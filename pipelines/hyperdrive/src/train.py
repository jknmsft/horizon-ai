import os
import json
import random
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from azureml.core import Run
from datetime import datetime
import matplotlib.pyplot as plt
from azureml.core import Dataset
from standalone.algo import models
from tensorflow.keras.callbacks import Callback

def main():

    # Set the seed
    random.seed(a=12)

    # Logging configuration
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    # Create the arguments parser
    parser = argparse.ArgumentParser(prog='HyperDrive')

    parser.add_argument('--data', type=str, dest='data', required=True, help='Data folder')
    parser.add_argument('--csv-file', type=str, dest='csv_file_path', required=False, help='File path of the CSV')
    parser.add_argument('--latent-dim-1', type=int, dest='latent_dim_1', default=5, help='Number of filters in first CNN layer')
    parser.add_argument('--latent-dim-2', type=int, dest='latent_dim_2', default=5, help='Number of filters in second CNN layer')
    parser.add_argument('--kernel-size', type=int, dest='kernel_size', default=3, help='Kernel size')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='Learning rate')
    parser.add_argument('--alpha', type=float, dest='alpha', default=0.0, help='Alpha parameter')
    parser.add_argument('--T', type=int, dest='T', default=72, help='T')
    parser.add_argument('--epochs', type=int, dest='epochs', default=100, help='Number of epochs')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Find the snapshot directory
    snapshot_dir = Path(__file__).parent.absolute()

    with Run.get_context() as current_run:

        # Create a callback class for Keras
        class LogRunMetrics(Callback):
            # Callback at the end of every epoch
            def on_epoch_end(self, epoch, log):
                # Log a value repeated which creates a list
                current_run.log(name='Loss', value=log['val_loss'])

        if current_run.id.startswith('OfflineRun'):
            ########################
            # CODE RUNNING LOCALLY #
            ########################

            logging.info('Local run')
            
            csv_file_path = os.path.join('./data', 'consommationfichier_conso_LAX10.csv')
            # Read the data from CSV
            try:
                file_df = pd.read_csv(csv_file_path, parse_dates=['datetime'], index_col=['datetime'])
            except FileNotFoundError as file_ex:
                print(f'Cannot open {csv_file_path}')
                raise file_ex

            # Start the clock
            start_datetime = datetime.now()

            # Get the identifier
            if len(file_df['lclid'].unique()) > 1:
                identifier = file_df['lclid'].unique()
            else:
                identifier = file_df['lclid'].unique()[0]  

            # NOTE: For some reasons, the Hydro-Quebec code doesn't work with the `lclid` feature
            file_df.drop(columns=['lclid'], inplace=True)

            # Create a CNN model 
            cnn = models.CNN(lr=args.learning_rate,
                      T=args.T,     
                      alpha=args.alpha,
                      latent_dim_1=args.latent_dim_1,
                      latent_dim_2=args.latent_dim_2, 
                      kernel_size=args.kernel_size)

            # Split the dataset
            train_inputs, valid_inputs, test_inputs, y_scaler = \
                cnn.create_input(data=file_df, T=args.T, HORIZON=24, cols=list(file_df.columns))

            # Get the CNN model from the `cnn` object
            cnn_model = cnn.get_model(length=len(list(file_df.columns)))

            # Train the CNN
            history = cnn_model.fit(
                    train_inputs["X"],
                    train_inputs["target"],
                    batch_size=16,
                    epochs=args.epochs,
                    validation_data=(valid_inputs["X"], valid_inputs["target"]),
                    callbacks=[LogRunMetrics()],
                    verbose='auto')     

            # Save the TF-Keras model
            os.makedirs('./models', exist_ok=True)
            cnn_model.save(filepath=os.path.join('./models', identifier),
                           overwrite=True,
                           include_optimizer=True,
                           save_format='tf')

            # Display the learning curve
            os.makedirs('./images', exist_ok=True)
            plt.plot(history.history['loss'], label='Train')
            plt.ylim([-.001,.05])
            plt.yticks(np.arange(0, 21, step=5))
            plt.plot(history.history['val_loss'], label='Validate')
            plt.xlabel('Época')
            plt.ylabel('Error')
            plt.ylim([-.001,.05])
            plt.title('CNN-' + identifier)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join('./images', identifier + '-' + 'training-curve.jpg'), format='jpg', dpi=150)
            plt.close()

            # Stop the clock
            end_datetime = datetime.now()

            # Display the training time
            duration = end_datetime - start_datetime
            print(f'Duration: {duration}')
            print(f"Model Size: {os.path.getsize(os.path.join('./models', identifier, 'saved_model.pb'))} bytes")
        else:
            #########################
            # CODE RUNNING REMOTELY #
            #########################

            logging.info('Remote Run')

            # Get the workspace from the run context
            ws = current_run.experiment.workspace

            # Get the default datastore from the workspace object
            default_dstore = ws.get_default_datastore()

            # DEBUG
            DATA_DIR = args.data
            CSV_FILE = args.csv_file_path
            files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR)]
            basenames = [os.path.basename(f) for f in files]
            
            # Read the data from CSV
            if CSV_FILE in basenames:             
                try:
                    logging.info(f'Read {os.path.join(DATA_DIR, CSV_FILE)}')
                    file_df = pd.read_csv(os.path.join(DATA_DIR, CSV_FILE), parse_dates=['datetime'], index_col=['datetime'])
                except FileNotFoundError as file_ex:
                    print(f'Error reading CSV file at {os.path.join(DATA_DIR, CSV_FILE)}')
                    raise file_ex
            else:
                try:
                    # Choose a random file in basenames
                    idx = random.randrange(len(basenames))
                    logging.info(f'Read {files[idx]}')
                    file_df = pd.read_csv(files[idx], parse_dates=['datetime'], index_col=['datetime'])
                except FileNotFoundError as file_ex:
                    print(f'Error reading CSV file at {files[idx]}')
                    raise file_ex

            # Get the identifier
            if len(file_df['lclid'].unique()) > 1:
                identifier = file_df['lclid'].unique()
            else:
                identifier = file_df['lclid'].unique()[0]  

            # NOTE: Hydro-Quebec code doesn't work with the `lclid` feature
            file_df.drop(columns=['lclid'], inplace=True)

            # Create the CNN model
            cnn = models.CNN(lr=args.learning_rate,
                                     T=args.T, 
                                     alpha=args.alpha,
                                     latent_dim_1=args.latent_dim_1,
                                     latent_dim_2=args.latent_dim_2, 
                                     kernel_size=args.kernel_size)

            # Split the dataset
            train_inputs, valid_inputs, test_inputs, y_scaler = \
                            cnn.create_input(data=file_df, T=args.T, HORIZON=24, cols=list(file_df.columns))

            # Get the CNN model from the `cnn` object
            cnn_model = cnn.get_model(length=len(list(file_df.columns)))

            # Train the CNN
            history = cnn_model.fit(
                        train_inputs["X"],
                        train_inputs["target"],
                        batch_size=16,
                        epochs=args.epochs,
                        validation_data=(valid_inputs["X"], valid_inputs["target"]),
                        callbacks=[LogRunMetrics()],
                        verbose='auto')     

            # Save the TF-Keras model
            os.makedirs('./models', exist_ok=True)
            cnn_model.save(filepath=os.path.join('./models', identifier),
                           overwrite=True,
                           include_optimizer=True,
                           save_format='tf')

            # Display the learning curve
            os.makedirs('./images', exist_ok=True)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validate')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('CNN-' + identifier)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join('./images', identifier + '-' + 'training-curve.jpg'), format='jpg', dpi=150)
            plt.close()

            # Log the learning curve image
            try:
                current_run.log_image(name='Learning Curve', 
                                            path=os.path.join('./images', identifier + '-' + 'training-curve.jpg'),
                                            description='CNN Learning Curve')
            except BaseException as bex:
                print('Error logging learning curve to run')
                raise bex

            # Upload the `./models/identifier/` folder to the run
            try:
                current_run.upload_folder(
                                name='models', 
                                path=os.path.join('./models', identifier),
                                datastore_name=None)
            except BaseException as bex:
                print('Error uploading model into run')
                raise bex

            ## Upload the `./models/model_name/` folder to the defaut datastore
            # try:
            #     default_dstore.upload(src_dir=os.path.join('./models', model_name),
            #                       target_path=f'/data/hydroqc/prs/train/models/{model_name}',
            #                       overwrite=True)
            # except BaseException as bex:
            #     print(f'Error uploading {file_name} to datastore')
            #     child_run.fail()
            #     raise bex

            # Tag the run
            tags = {'model': 'CNN', 'lclid': identifier, 'id': current_run.id}
            current_run.set_tags(tags=tags)                          

if __name__ == '__main__':
    main()    