import os
import argparse
import pandas as pd
from pathlib import Path
from azureml.core import Run
from datetime import datetime
from sklearn.pipeline import Pipeline
from standalone.algo.utils.data import *

# NOTE: azureml_user package is only available on AzureML compute
from azureml_user.parallel_run import EntryScript

# Setup run
current_run = None

# NOTE: The `init` method is called once from each
# NOTE: worker process on every node on which the job is running
def init():
    
    global current_run
    current_run = Run.get_context()

    # Enable PRS logging
    entry_script = EntryScript()
    logger = entry_script.logger

    # DEBUG
    logger.info('Calling init()')

# NOTE: The `run` method is called for each mini_batch instance 
# NOTE: For a FileDataset, the mini-batch returns the file paths in the datastore
# NOTE: For a TabularDataset, the mini-batch returns a pandas dataframe        
def run(mini_batch):

    # Enable PRS logging
    entry_script = EntryScript()
    logger = entry_script.logger
    logger.info('Calling run()')

    # Find the snapshot directory
    snapshot_dir = Path(__file__).parent.absolute()     

    # Get the workspace from the run context
    ws = current_run.experiment.workspace

    # Get the default datastore from the workspace object
    default_dstore = ws.get_default_datastore()

    # Section 1: Setup output directory and the results list
    os.makedirs('./outputs', exist_ok=True)
    result_list = []

    # Loop through each file in the mini-batch
    # NOTE: The number of files in each batch is controlled by the mini_batch_size parameter of ParallelRunConfig
    for idx, csv_file_path in enumerate(mini_batch):
        result = {}

        # Start the clock
        start_datetime = datetime.now()   

        # Get the filename from the csv_file_path
        file_name = os.path.basename(csv_file_path)
        base_name = file_name[:-4]

        # Read the data from CSV
        try:
            file_df = pd.read_csv(csv_file_path)
        except FileNotFoundError as file_ex:
            print(f'Cannot open {csv_file_path}')
            raise file_ex                           
        
        child_run = None
        try:
            # Create a child run
            # NOTE: We could optimize this with create_children method
            child_run = current_run.child_run()
            
            # Drop the two first columns
            file_df.drop(columns=file_df.columns[:2], inplace=True)
        
            # Apply one-hot encoding to the categorical features
            # NOTE: DateTime and LCLid are `reserved`
            file_df = pd.get_dummies(file_df, columns=['precipType', 'icon', 'summary'], prefix='cat')  
        
            # Remove spaces in feature names
            file_df.columns = [col.lower().replace(' ', '-') for col in file_df.columns]      
        
            # Convert `datetime` feature to a datetime type
            file_df['datetime'] = pd.to_datetime(file_df['datetime'])

            # Set `datetime` feature as the dataframe index
            file_df.set_index(keys=['datetime'], inplace=True)  
        
            # Make a pipeline with the steps DateTimeFeatures and CyclicalDateTimeFeatures
            transforms = Pipeline(
                            steps=[
                                    # Must create the date/time features before encoding
                                    ("date_time_features", DateTimeFeatures()),
                                    ("cylical_date_time_features", CyclicalDateTimeFeatures()),
                                    #("holidays_features", HolidaysFeatures()),
                                  ])      

            # Apply the data transformations
            file_df = transforms.transform(file_df)  

            # Removes consumption values at 0
            if not (file_df['consumption'] == 0).sum():
                # Replace zero-valued `consumption` for NAN
                file_df['consumption'].replace(to_replace=0, value=np.nan, inplace=True)

                # Replace NAN values with interpolation values
                file_df['consumption'].interpolate(method='time', inplace=True)

            # Sort the index in ascending order
            file_df.sort_index(inplace=True)
            
            # Save the CSV into `./outputs/csv/`` folder
            os.makedirs(os.path.join('./outputs', 'csv'), exist_ok=True)
            file_df.to_csv(os.path.join('./outputs', 'csv', file_name))

            # Find the identifier for each file
            if len(file_df['lclid'].unique()) > 1:
                identifier = file_df['lclid'].unique()
            else:
                identifier = file_df['lclid'].unique()[0]  

            # Tag the child run
            tags = {'Filename': file_name, 'LCLid': identifier}
            child_run.set_tags(tags=tags)

            # Upload the CSV file in the child run
            # NOTE: Not optimal because we already upload files into datastore
            try:
                child_run.upload_file(name=file_name, path_or_stream=os.path.join('./outputs', 'csv', file_name))
            except BaseException as bex:
                print(f'Error uploading {file_name} to the run')
                raise bex

            # Upload the CSV files to the default datastore
            try:
                default_dstore.upload(src_dir=os.path.join('./outputs', 'csv'),
                                  target_path='/data/hydroqc/clean/',
                                  overwrite=True)
            except BaseException as bex:
                print(f'Error uploading {file_name} to datastore')
                child_run.fail()
                raise bex

            # Mark the child run as completed
            child_run.complete()
            
            # Stop the clock
            end_datetime = datetime.now()

            # Add data to output
            result['Filename'] = file_name
            result['Bytes'] = os.path.getsize(csv_file_path)
            result['StartDate'] = str(start_datetime)
            result['EndDate'] = str(end_datetime)
            result['Duration'] = str(end_datetime - start_datetime)
            result['RunID'] = str(child_run.id)
            result['Status'] = child_run.get_status()

            # Create the final dictionary for output
            result_list.append(result)
        except BaseException as bex:
            # If the child run status isn't completed, mark as failed
            if child_run and child_run.get_status() != 'Completed':
                child_run.fail()

            raise bex

    # Data returned by this function will be available in parallel_run_step.txt
    return pd.DataFrame(result_list)