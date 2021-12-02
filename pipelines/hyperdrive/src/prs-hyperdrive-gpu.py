import os
import json
import argparse
import pandas as pd
from pathlib import Path
from azureml.core import Run
from datetime import datetime
import matplotlib.pyplot as plt
from azureml.core import Dataset
from standalone.algo import models
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.model import Model
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import choice
from azureml.core.compute import AmlCompute
from tensorflow.keras.callbacks import Callback
from azureml.train.hyperdrive import HyperDriveConfig
from azureml.train.hyperdrive import BayesianParameterSampling
from azureml.train.hyperdrive.runconfig import PrimaryMetricGoal
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

# NOTE: azureml_user package is only available on AzureML compute
from azureml_user.parallel_run import EntryScript

# Setup run
current_run = None

# NOTE: The `init` method is called once from each
# NOTE: worker process on every node on which the job is running
def init():
    
    global current_run

    # Create the run context
    current_run = Run.get_context()

    # Enable PRS logging
    entry_script = EntryScript()
    logger = entry_script.logger

    # NOTE: Debug message will show in files under logs/user in the portal
    logger.info('Calling init()')

# NOTE: The `run` method is called for each mini_batch instance  
# NOTE: For a FileDataset, the mini-batch returns the file path in the datastore
# NOTE: For a TabularDataset, the mini-batch returns a pandas dataframe   
def run(mini_batch):
    
    # Enable PRS logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Calling run()')
    logger.info(f'mini_batch: {mini_batch}')

    # Find the snapshot directory
    snapshot_dir = Path(__file__).parent.absolute()
    logger.info(f'Snapshot directory: {snapshot_dir}')
    logger.info(f'Files: {os.listdir(snapshot_dir)}')

    # Read the parameters.json file
    with open(os.path.join(snapshot_dir, 'parameters.json'), mode='r') as f:
        try:
            params = json.load(f)
        except BaseException as bex:
            print(f'Cannot open parameters.json')
            raise bex

    # Get the workspace from the run context
    ws = current_run.experiment.workspace

    # Get the default datastore from the workspace object
    default_dstore = ws.get_default_datastore()

    # Get the file dataset
    try:
        clean_fds = Dataset.get_by_name(workspace=ws, name='clean_fds')
        input_dataset = clean_fds.as_named_input(name='clean_fds')
    except NameError as name_ex:
        print(f"Can't find the registered dataset")
        raise name_ex

    # Setup `outputs` folder and the PRS result list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # Create the training environment
    # NOTE: We can register the environment in Workspace
    try:
        train_gpu_env = Environment(name="train-gpu-env").from_conda_specification(
                                                    name='train-gpu-env',
                                                    file_path=os.path.join(snapshot_dir, 'train-gpu-env.yml')) 
    except BaseException as bex:
        print('Failed to create environment')
        raise bex

    # Create the parameters search space
    try:
        ps = {'latent-dim-1': choice(*params['HYPER_PARAMS']['--latent-dim-1']),
              'latent-dim-2': choice(*params['HYPER_PARAMS']['--latent-dim-2']),
              'kernel-size': choice(params['HYPER_PARAMS']['--kernel-size']),
              'learning-rate': choice(*params['HYPER_PARAMS']['--learning-rate']),
              'alpha': choice(*params['HYPER_PARAMS']['--alpha']), 
              'T': choice(*params['HYPER_PARAMS']['--T'])}  
    except BaseException as bex:
        print('Error creating parameters search space')
        raise bex  

    # Choose the sampling method
    parameter_sampling = BayesianParameterSampling(parameter_space=ps)

    # Specify the early termination policy
    # NOTE: BayesianParameterSampling doesn't support early termination
    early_termination_policy = None              

    # Loop through each file in the mini-batch
    # NOTE: The number of files in each batch is controlled by the mini_batch_size 
    # parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(mini_batch):
        result = {}

        # Start the clock
        start_datetime = datetime.now()    

        # Get the filename from the csv_file_path
        file_name = os.path.basename(csv_file_path)
        base_name = file_name[:-4]

        try:
            # cluster_name = f'hd-cluster-{idx + 1}'
            cluster_name = f'hd{idx + 1}-K80-gpu-ci'
            compute_target = ws.compute_targets[cluster_name]
        except BaseException as bex:
            print('Cannot find compute target')
            raise bex

        try:
            # Create the run configuration
            hd_src = ScriptRunConfig(source_directory=os.path.join(snapshot_dir),
                      script='train-gpu.py',
                      arguments=['--data', input_dataset.as_mount(path_on_compute='/tmp/clean_fds/'), '--csv-file', file_name, '--epochs', 20],
                      compute_target=compute_target,
                      environment=train_gpu_env)
        
            # Create the hyperdrive run configuration
            hyperdrive_config = HyperDriveConfig(run_config=hd_src,
                                     hyperparameter_sampling=parameter_sampling,
                                     policy=early_termination_policy,
                                     primary_metric_name='Loss',
                                     primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                     max_total_runs=120,
                                     max_concurrent_runs=18)

            # Submit hyperdrive run
            hd_run = current_run.submit_child(config=hyperdrive_config)
            
            # Tag the child run
            hd_run.set_tags(tags={'File': file_name})

            # Store results
            # duration = end_datetime - start_datetime
            # result['start_datetime'] = start_datetime
            # result['end_datetime'] = end_datetime
            # result['duration'] = duration
            result['run.name'] = hd_run.name
            result['run.id'] = hd_run.id

            result_list.append(result)

        except BaseException as bex:
            if hd_run and hd_run.get_status() != 'Completed':
                hd_run.cancel()
                hd_run.parent.fail()
                
            raise bex            

    return pd.DataFrame(result_list)