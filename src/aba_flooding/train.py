import model as md
import pandas as pd
import geopandas as gpd
import geo_utils as gu
import matplotlib.pyplot as plt
import os
from preprocess import load_saved_data

"""
From preprocess.py take the load_saved_data and load in the parquet files named by stations
columns for files are: 
        '{station}_WOG_{soil_type}'
        '{station}_{soil_type}_observed"
        '{station}_{soil_type}_TTE'
        '{station}_{soil_type}_duration'
Station er med på alle så husk at juster efter det
"""


# Import functions from the new preprocess module

def load_models(path):
    """
    Load trained flood models from a pickle file.
    
    Parameters:
    -----------
    path : str
        Path to the saved model file
        
    Returns:
    --------
    FloodModel : Trained flood model
    """
    # TODO:
    pass

def train_all_models():
    """
    Train survival models for all specified soil types.
    
    Parameters:
    -----------
    soiltypes : list
        List of soil types to train models for
        
    Returns:
    --------
    FloodModel : Trained flood model
    """
    # Find all files in data/processed/
    processed_data_path = os.path.join(os.getcwd(), "data/processed/")
    station_names = []

    # Check if directory exists
    if os.path.exists(processed_data_path):
        # List all files in the directory
        files = os.listdir(processed_data_path)
        
        # Extract station names (part before the '-')
        for file in files:
            if file.endswith('.parquet'):
                station_name = file.split('_')[2].strip()
                if station_name and station_name not in station_names:
                    station_names.append(station_name)
    else:
        print(f"Directory {processed_data_path} does not exist")

    print(f"Found {len(station_names)} stations: {station_names}")

    floodModel = md.FloodModel()


    # For each station
    for station in station_names:
        # Load the data for the station
        station_data_path = os.path.join(processed_data_path, f"survival_data_{station}.parquet")   
        survival_df = load_saved_data(station_data_path)

        # extract the soil types from the dataframe
        soiltypes = []
        for column in list(survival_df.columns[1:]):
            # Check if the soil type is already in the flood model
            soiltype = column.split('_')[1]
            if soiltype != 'WOG':
                soiltypes.append(soiltype)
            if soiltype not in floodModel.available_soil_types and soiltype != 'WOG':
                floodModel.available_soil_types.append(soiltype)

        # Create and train models for the floodmodel (station)
        floodModel.add_station(station, survival_df, soiltypes)

    floodModel.save()

    return floodModel
    

from lifelines import KaplanMeierFitter

if __name__ == "__main__":
    # Example usage
    
    flood_model = train_all_models()
    print("Flood model trained successfully.")
    print("Available soil types:", flood_model.available_soil_types)
    print("Trained models:", list(flood_model.models.keys()))
    
    # Plot all models
    #flood_model.plot_all(save=True)

