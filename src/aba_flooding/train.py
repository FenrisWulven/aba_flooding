import model as md
import pandas as pd
import geopandas as gpd
import geo_utils as gu
import matplotlib.pyplot as plt
import os

# Import functions from the new preprocess module
from preprocess import load_process_data, preprocess_data_for_survival, load_saved_data

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
    import pickle
    with open(path, 'rb') as f:
        flood_model = pickle.load(f)
    return flood_model

def train_all_models(soiltypes):
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
    
    processed_data_path = "data/processed/survival_data.csv"
    
    if os.path.exists(processed_data_path):
        print(f"Loading preprocessed survival data from {processed_data_path}")
        # Use the function from the preprocess module
        survival_dfs = load_saved_data(processed_data_path)
        
        # Load raw data for the model training
        df = load_process_data()
        
    else:
        print("Preprocessing data and saving results...")
        # Load and process data using the function from preprocess module
        df = load_process_data()
        
        # Get the soil types we actually have data for in our absorption dictionary
        available_soiltypes = list(df.filter(regex=r'.*observed$').columns)
        available_soiltypes = [col.replace('observed', '') for col in available_soiltypes]
        
        print(f"Data available for soil types: {available_soiltypes}")
        
        # Filter to only include soil types that we have data for
        valid_soiltypes = [st for st in soiltypes if st in available_soiltypes]
        if not valid_soiltypes:
            print("Warning: None of the requested soil types have data. Using all available soil types.")
            valid_soiltypes = available_soiltypes
        
        # Preprocess data for survival analysis using the function from preprocess module
        survival_dfs = preprocess_data_for_survival(df, valid_soiltypes)
        
        # Save preprocessed data 
        from preprocess import save_preprocessed_data
        save_preprocessed_data(survival_dfs, processed_data_path)
    
    # Initialize FloodModel with all requested soil types (even those without data)
    floodModel = md.FloodModel()
    floodModel.soil_types = soiltypes
    
    # Train the FloodModel directly with survival dataframes
    floodModel.train(data=df, survival_dfs=survival_dfs, 
                     duration_column='duration', event_column='observed')
    
    #print(f"Successfully trained models for {len(floodModel.available_soil_types)} soil types")
    
    # Save the trained model
    import pickle
    os.makedirs(os.path.dirname(os.path.join(os.getcwd(), "models/")), exist_ok=True)
    with open(os.path.join(os.getcwd(), "models/flood_models.pkl"), 'wb') as f:
        pickle.dump(floodModel, f)
    print(f"Saved trained model to models/flood_models.pkl")
    
    # Return just the flood model
    return floodModel

from lifelines import KaplanMeierFitter

if __name__ == "__main__":
    # Example usage
    soil_types = ["DG - Meltwater gravel", "DS - Meltwater sand"]
    
    flood_model = train_all_models(soil_types)
    print("Flood model trained successfully.")
    print("Available soil types:", flood_model.available_soil_types)
    print("Trained models:", list(flood_model.models.keys()))
    
    # Plot all models
    #flood_model.plot_all(save=True)

