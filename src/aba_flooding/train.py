import model as md
import pandas as pd
import geopandas as gpd
import geo_utils as gu


# QUick Intro:
# Goal: 
# create a column that indicates "is_flood" for each soil type in the list of soil types
# Then create a dict of survival models that correspond to soiltypes.
# this dict of trained models will be passed to the flood_layer to give it directions for its heatmap.s

def load_process_data():
    # TODO

    df = pd.read_csv("data/raw/Regn2004-2025.csv", sep=";")
    df['Dato'] = pd.to_datetime(df['Dato'], format='%d.%m.%Y')

    df['Tid'] = df.apply(lambda x: int(x['Tid'].split(':')[0]), axis=1)
    df['datetime'] = df['Dato'] + pd.to_timedelta(df['Tid'], unit='h')
    absorbtions = {'DG - Meltwater gravel': 0.1, 'DS - Meltwater sand': 0.2}

    for soil_type in absorbtions.keys():
        df[f'WOG_{soil_type}'] = 0.0
        
        # Apply cumulative calculation using expanding window
        for curr_soil, rate in absorbtions.items():  # Changed variable name
            # Create temporary series for calculation
            wog = pd.Series(0.0, index=df.index)
            
            # Loop is necessary because each calculation depends on previous result
            for i in range(1, len(df)):
                wog.iloc[i] = max(0, wog.iloc[i-1] * (1 - rate) + df['Nedbor'].iloc[i])
                
            # Assign back to dataframe
            df[f'WOG_{curr_soil}'] = wog  # Use new variable name

        df.dropna(inplace=True)
        
        # Rest of code as before
        df[f"{soil_type}observed"] = (df[f"WOG_{soil_type}"] > 5).astype(int)
        

        dry_spells = []
        current_duration = 0
        censored = False  # True if the last dry spell was ongoing at the end of the dataset
        durations = []

        for idx, row in df.iterrows():
            if row[f"{soil_type}observed"] == 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    dry_spells.append({"duration": current_duration, "observed": 1})
                    current_duration = 0
            durations.append(current_duration)
        # Handle censoring (if the dataset ends with a dry spell)
        if current_duration > 0:
            dry_spells.append({"duration": current_duration, "observed": 0})

        df[f'{soil_type}duration'] = durations
        print("columns: ", df.columns)

    return df

def gather_soil_types(sediment_geo_json):
    # Gets all possible soil types and matches them with the available data.
    # TODO
    pass

def train_all_models(soiltypes):
        # Will train a model for each soiltype (depending on availability of data)
    df = load_process_data()

    print("Available columns:", df.columns.tolist())
    
    # Initialize FloodModel with the correct soil types
    floodModel = md.FloodModel()
    floodModel.soil_types = soiltypes  # Make sure FloodModel uses the same soil types

    # Pass the base column names without prefixes
    floodModel.train(df, duration_column='duration', event_column='observed')

    # load geodata
    sediment_data = gu.load_terrain_data("Sediment.geojson")
    sediment_data = floodModel.predict_proba(sediment_data, year=1)
    
    return floodModel

