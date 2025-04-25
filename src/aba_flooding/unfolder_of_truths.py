
#=================================================================================================#
# This script processes JSON files containing precipitation data, extracts relevant information,  #
# and saves the data into a Parquet file. It also logs the processing steps and statistics.       #
#=================================================================================================#

import os
import json
import pandas as pd
from datetime import datetime
import glob
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("precipitation_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_json_files(folder_path):
    # Get all json files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Create a dictionary to store all data
    all_data = {}
    total_records = 0
    
    # Process each file with a progress bar
    for file_path in tqdm(json_files, desc="Processing JSON files"):
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        
        try:
            # Load the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            file_records = 0
            # Extract the data points
            for item in data['data']:
                # Get timestamp, station ID, and value
                timestamp = item['properties']['observed']
                station_id = item['properties']['stationId']
                value = item['properties']['value']
                
                # Convert timestamp to datetime object
                timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                
                # Store in dictionary: all_data[timestamp][station_id] = value
                if timestamp not in all_data:
                    all_data[timestamp] = {}
                
                all_data[timestamp][station_id] = value
                file_records += 1
            
            total_records += file_records
            logger.info(f"Extracted {file_records} records from {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
    logger.info(f"Total records processed: {total_records}")
    logger.info("Converting to DataFrame...")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_data, orient='index')
    
    # Sort index (timestamps) chronologically
    df.sort_index(inplace=True)
    
    logger.info(f"DataFrame created with shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    folder_path = "/Users/maks/Documents/GitHub/aba_flooding/dmi_data_daily/parameter_batches/precip_past1h_test"
    
    logger.info("Starting precipitation data processing")
    
    # Process the files
    precipitation_df = process_json_files(folder_path)
    
    # Get some stats
    memory_usage_mb = precipitation_df.memory_usage(deep=True).sum() / 1048576
    num_stations = len(precipitation_df.columns)
    num_timestamps = len(precipitation_df)
    
    # Save to Parquet
    logger.info("Saving data to Parquet file...")
    precipitation_df.to_parquet("precipitation_data_test.parquet")
    
    logger.info(f"Processing complete. Data saved to precipitation_data_test.parquet")
    logger.info(f"DataFrame shape: {precipitation_df.shape}")
    logger.info(f"Number of timestamps: {num_timestamps}")
    logger.info(f"Number of stations: {num_stations}")
    logger.info(f"DataFrame memory usage: {memory_usage_mb:.2f} MB")
    logger.info(f"Date range: {precipitation_df.index.min()} to {precipitation_df.index.max()}")
    
    print("\nSample of the data:")
    print(precipitation_df.head())