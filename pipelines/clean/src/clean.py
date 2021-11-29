import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from standalone.algo.utils.data import *

def main():

    # Logging configuration
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    # Create the arguments parser
    parser = argparse.ArgumentParser(prog='Clean')

    parser.add_argument('--data', type=str, dest='data', required=True, help='Data folder')
    parser.add_argument('--csv-file', type=str, dest='csv_file_path', required=False, help='File path of the CSV')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # DEBUG
    logging.info('Local run')
            
    DATA_DIR = args.data
    CSV_FILE = args.csv_file_path        
    csv_file_path = os.path.join(DATA_DIR, CSV_FILE)
    logging.info(f'csv_file_path: {csv_file_path}')

    # Read the data from CSV
    try:
        file_df = pd.read_csv(csv_file_path)
    except FileNotFoundError as file_ex:
        logging.exception(f'File {CSV_FILE} not found')
        raise file_ex

    # Start the clock
    start_datetime = datetime.now()      

    # Get the filename from the csv_file_path
    file_name = os.path.basename(csv_file_path)
    base_name = file_name[:-4]                   
            
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

    # Stop the clock
    end_datetime = datetime.now()  

    # Display information about dataframe
    file_df.info()

    # Display the computation time
    duration = end_datetime - start_datetime
    logging.info(f'Duration: {duration}')
    logging.info(f"File size: {os.path.getsize(os.path.join('./outputs', 'csv', file_name))} bytes")      

if __name__ == '__main__':
    main()    