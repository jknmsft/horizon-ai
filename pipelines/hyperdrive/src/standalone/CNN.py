########
# GOAL #
########

Adapt the CNN code with the custom script of the many models accelerator

Step 1:
=======

The raw data reside in Azure Storage Account:
    - Connect this Azure Storage Account to the AzureML Workspace
    
Data transformations:    
    - For each file in training1 folder:
        - Load file into a pandas dataframe
        - Remove the index and the unnamed.0 columns
        - Remove zero value of the consumption feature
        - Capitalize columns name
        - Apply the feature engineering 
        - Save the file to CSV

Dataset:
    - Create a FileDataset with the cleaned data
    - Register dataset
    
Training:
    - Train the CNN model (as-is, no fine-tuning)
    - Hyperdrive tuning
    - Register models into Model Registry
    
Metrics:
    - Compute training time metrics
    -         




class CNN():
    def __init__(self, lr, T, alpha, latent_dim_1, latent_dim_2, kernel_size, padding='causal', activation='relu', dilation_rate=1):
        self.lr = lr
        self.T = T
        self.alpha = alpha
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dilation_rate = dilation_rate

    def get_model(self, length, HORIZON=24):
        model = Sequential()
        model.add(
            Conv1D(
                filters=self.latent_dim_1,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation=self.activation,
                dilation_rate=self.dilation_rate,
                input_shape=(self.T, length),
                kernel_regularizer=regularizers.l2(self.alpha),
                bias_regularizer=regularizers.l2(self.alpha),
            )
        )
        if self.latent_dim_2:
            model.add(
                Conv1D(
                    filters=self.latent_dim_2,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    dilation_rate=2,
                    kernel_regularizer=regularizers.l2(self.alpha),
                    bias_regularizer=regularizers.l2(self.alpha),
                )
            )

        model.add(Flatten())
        model.add(
            Dense(
                units=HORIZON,
                activation='linear',
                kernel_regularizer=regularizers.l2(self.alpha),
                bias_regularizer=regularizers.l2(self.alpha),
            )
        )

        optimizer = RMSprop(lr=self.lr)
        model.compile(optimizer=optimizer, loss="mse")

        return model
        
# Create a CNN model
cnn = CNN(lr=params["SCRIPT_PARAMS"]['--learning-rate'],
          T=params["SCRIPT_PARAMS"]['--T'], 
          alpha=params["SCRIPT_PARAMS"]['--alpha'],
          latent_dim_1=params["SCRIPT_PARAMS"]['--latent-dim-1'],
          latent_dim_2=params["SCRIPT_PARAMS"]['--latent-dim-2'], 
          kernel_size=params["SCRIPT_PARAMS"]['--kernel-size']
)        


from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication

#interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", force=True)

try:
    ws = Workspace.get(
        name='mlgeek-ws',
        subscription_id='e40e1658-df4f-4dfc-b90f-158e55336daa',
        resource_group='mlgeek-rg')

    # Write the details of the workspace to a configuration file
    ws.write_config()
    print("Workspace configuration succeeded.")
except Exception as e:
    print('Workspace not accessible.')
    print('Error: {}'.format(e))
    
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name='cpu-cluster')
    print('Found an existing cluster, using it instead.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D13_V2',
                                                           min_nodes=0,
                                                           max_nodes=20)
    cpu_cluster = ComputeTarget.create(workspace=ws, name='cpu-cluster', provisioning_configuration=compute_config)
    cpu_cluster.wait_for_completion(show_output=True)
    
# Connect to default datastore
datastore = ws.get_default_datastore()

# Upload train data
ds_train_path = target_path + 'train'
datastore.upload(src_dir=train_path, target_path=ds_train_path, overwrite=True)

# Upload inference data
ds_inference_path = target_path + 'inference'
datastore.upload(src_dir=inference_path, target_path=ds_inference_path, overwrite=True)

""" blob_datastore_name = "automl_many_models"
container_name = "automl-sample-notebook-data"
account_name = "automlsamplenotebookdata" """

""" from azureml.core import Datastore

datastore = Datastore.register_azure_blob_container(
    workspace=ws, 
    datastore_name=blob_datastore_name, 
    container_name=container_name,
    account_name=account_name,
    create_if_not_exists=True
)

if 0 < dataset_maxfiles < 11973:
    ds_train_path = 'oj_data_small/'
    ds_inference_path = 'oj_inference_small/'
else:
    ds_train_path = 'oj_data/'
    ds_inference_path = 'oj_inference/' """
    
from azureml.core.dataset import Dataset

# Create file datasets
ds_train = Dataset.File.from_files(path=datastore.path(ds_train_path), validate=False)
ds_inference = Dataset.File.from_files(path=datastore.path(ds_inference_path), validate=False)

# Register the file datasets
dataset_name = 'oj_data_small' if 0 < dataset_maxfiles < 11973 else 'oj_data'
train_dataset_name = dataset_name + '_train'
inference_dataset_name = dataset_name + '_inference'
ds_train.register(workspace=ws, name=train_dataset_name, create_new_version=True)
ds_inference.register(workspace=ws, name=inference_dataset_name, create_new_version=True)            
