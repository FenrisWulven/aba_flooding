import model as md
import pandas as pd
import geopandas as gpd
import geo_utils as gu
import matplotlib.pyplot as plt
import perculation_mapping as pm
import os

# QUick Intro:
# Goal: 
# create a column that indicates "is_flood" for each soil type in the list of soil types
# Then create a dict of survival models that correspond to soiltypes.
# this dict of trained models will be passed to the flood_layer to give it directions for its heatmap.s

def load_process_data():
    df = pd.read_csv("data/raw/Regn2004-2025.csv", sep=";")
    df['Dato'] = pd.to_datetime(df['Dato'], format='%d.%m.%Y')

    df['Tid'] = df.apply(lambda x: int(x['Tid'].split(':')[0]), axis=1)
    df['datetime'] = df['Dato'] + pd.to_timedelta(df['Tid'], unit='h')
    absorbtions = gather_soil_types(pm.percolation_rates_updated)

    for soil_type, rate in absorbtions.items():
        df[f'WOG_{soil_type}'] = 0.0
    
        # Create temporary series for calculation
        wog = pd.Series(0.0, index=df.index)
        
        # Loop is necessary because each calculation depends on previous result
        for i in range(1, len(df)):
            wog.iloc[i] = max(0, wog.iloc[i-1] * (1 - rate) + df['Nedbor'].iloc[i])
            
        # Assign back to dataframe
        df[f'WOG_{soil_type}'] = wog

        df.dropna(inplace=True)
        
        df[f"{soil_type}observed"] = (df[f"WOG_{soil_type}"] > 10).astype(int)

        dry_spells = []
        current_duration = 0
        censored = False  # True if the last dry spell was ongoing at the end of the dataset
        durations = []
        in_wet_state = False  # Track if we're currently in a wet period

        for idx, row in df.iterrows():
            if row[f"{soil_type}observed"] == 0:  # Dry period
                if in_wet_state:  # Transition from wet to dry
                    in_wet_state = False
                current_duration += 1
            else:  # Wet period
                if not in_wet_state:  # Only count transition from dry to wet as an event
                    in_wet_state = True
                    if current_duration > 0:
                        # End of a dry spell
                        dry_spells.append({"duration": current_duration, "observed": 1})
                        current_duration = 0
                # If already in wet state, ignore this wet period (treating as same event)
            
            durations.append(current_duration)
        # Handle censoring (if the dataset ends with a dry spell)
        if current_duration > 0:
            dry_spells.append({"duration": current_duration, "observed": 0})

        df[f'{soil_type}duration'] = durations

    return df

def gather_soil_types(purculation_mapping):
    # Take perculation Keys and the min and max / 2 and add to a dict
    # soil_types = {}
    soil_types = {}
    for key, value in purculation_mapping.items():
        min = 0.000001 if value['min'] == 0 else value['min']
        max = 0.999999 if value['max'] == 1 else value['max']
            
        soil_types[key] = (min + max) / 2
    return soil_types

def preprocess_data_for_survival(df, soil_types):
    """
    Process data to create survival dataframes for each soil type.
    Treats consecutive flooding events as a single event.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed dataframe with observed columns for each soil type
    soil_types : list
        List of soil types to process
        
    Returns:
    --------
    dict : Dictionary mapping soil types to their respective survival dataframes
    """
    survival_dfs = {}
    available_soil_types = []
    
    # First, verify which soil types have data
    for soil_type in soil_types:
        column_observed = f"{soil_type}observed"
        if column_observed in df.columns:
            available_soil_types.append(soil_type)
        else:
            print(f"Warning: No data found for soil type '{soil_type}' - skipping")
    
    # Only process soil types that have data
    for soil_type in available_soil_types:
        dry_spells = []
        current_duration = 0
        column_observed = f"{soil_type}observed"
        in_wet_state = False  # Track if we're currently in a wet period
        
        for idx, row in df.iterrows():
            if row[column_observed] == 0:  # Dry period
                if in_wet_state:  # Transition from wet to dry
                    in_wet_state = False
                current_duration += 1
            else:  # Wet period
                if not in_wet_state:  # Only process transitions from dry to wet
                    in_wet_state = True
                    if current_duration > 0:  # End of a dry spell
                        dry_spells.append({
                            "duration": current_duration, 
                            "observed": 1  # Event occurred (spell ended)
                        })
                        current_duration = 0
                # If already in wet state, ignore this wet period (treating as same event)
        
        # Handle the last spell if ongoing
        if current_duration > 0:
            dry_spells.append({"duration": current_duration, "observed": 0})
        
        # Create proper survival dataframe
        survival_df = pd.DataFrame(dry_spells)
        
        # Print statistics for this soil type
        print(f"\nStatistics for {soil_type} (grouping consecutive events):")
        print(f"Number of dry spells: {len(survival_df)}")
        print(f"Average duration: {survival_df['duration'].mean()}")
        print(f"Max duration: {survival_df['duration'].max()}")
        
        # Store in dictionary
        survival_dfs[soil_type] = survival_df
    
    return survival_dfs

def train_all_models(soiltypes):
    """
    Train survival models for all specified soil types.
    
    Parameters:
    -----------
    soiltypes : list
        List of soil types to train models for
        
    Returns:
    --------
    FloodModel : Trained flood model (not a tuple anymore)
    """
    
    processed_data_path = "data/processed/survival_data.csv"
    
    if os.path.exists(processed_data_path):
        print(f"Loading preprocessed survival data from {processed_data_path}")
        survival_df_combined = pd.read_csv(processed_data_path)
        
        # Convert back to dictionary of dataframes
        survival_dfs = {}
        for soil_type in soiltypes:
            soil_data = survival_df_combined[survival_df_combined['soil_type'] == soil_type]
            if len(soil_data) > 0:
                survival_dfs[soil_type] = soil_data[['duration', 'observed']].reset_index(drop=True)
                
                # Print statistics for this soil type
                print(f"\nStatistics for {soil_type} (loaded from file):")
                print(f"Number of dry spells: {len(survival_dfs[soil_type])}")
                print(f"Average duration: {survival_dfs[soil_type]['duration'].mean()}")
                print(f"Max duration: {survival_dfs[soil_type]['duration'].max()}")
        
        # Load raw data for the model training
        df = load_process_data()
        
    else:
        print("Preprocessing data and saving results...")
        # Load and process data
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
        
        # Preprocess data for survival analysis
        survival_dfs = preprocess_data_for_survival(df, valid_soiltypes)
        
        # Save preprocessed data
        # First, ensure directory exists
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        # Combine all survival dataframes into one with soil type column
        survival_df_combined = pd.DataFrame()
        for soil_type, soil_df in survival_dfs.items():
            soil_df_copy = soil_df.copy()
            soil_df_copy['soil_type'] = soil_type
            survival_df_combined = pd.concat([survival_df_combined, soil_df_copy], ignore_index=True)
        
        # Save to CSV
        survival_df_combined.to_csv(processed_data_path, index=False)
        print(f"Saved preprocessed survival data to {processed_data_path}")
    
    # Initialize FloodModel with all requested soil types (even those without data)
    floodModel = md.FloodModel()
    floodModel.soil_types = soiltypes
    
    # Train the FloodModel directly with survival dataframes
    floodModel.train(data=df, survival_dfs=survival_dfs, 
                     duration_column='duration', event_column='observed')
    
    print(f"Successfully trained models for {len(floodModel.available_soil_types)} soil types")
    
    # Return just the flood model (not the tuple)
    return floodModel

from lifelines import KaplanMeierFitter

if __name__ == "__main__":
    # Example usage
    soil_types = ["DG - Meltwater gravel", "DS - Meltwater sand"]
    
    # Now you can uncomment to use the integrated model training
    flood_model = train_all_models(soil_types)
    print("Flood model trained successfully.")
    print("Available soil types:", flood_model.available_soil_types)
    print("Trained models:", list(flood_model.models.keys()))
    
    # Plot all models
    flood_model.plot_all(save=True)

