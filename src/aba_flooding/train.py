import src.aba_flooding.model 
import pandas as pd

# QUick Intro:
# Goal: 
# create a column that indicates "is_flood" for each soil type in the list of soil types
# Then create a dict of survival models that correspond to soiltypes.
# this dict of trained models will be passed to the flood_layer to give it directions for its heatmap.s

def load_process_data(soil_types):
    # TODO

    df = pd.read_csv("Regn2004-2025.csv", sep=";")
    df['Dato'] = pd.to_datetime(df['Dato'], format='%d.%m.%Y')
    df_temp = pd.read_csv("Temperatur 2004-2025.csv", sep=";")
    df_temp['Dato'] = pd.to_datetime(df_temp['Dato'], format='%d.%m.%Y')
    df = df.merge(df_temp, on=['Dato', 'Tid'], how='left')

    df['Tid'] = df.apply(lambda x: int(x['Tid'].split(':')[0]), axis=1)
    df['datetime'] = df['Dato'] + pd.to_timedelta(df['Tid'], unit='h')
    absorbtions = {'clay': 0.1, 'sand': 0.1, 'silt': 0.1, 'peat': 0.1, 'loam': 0.1}

    for soil_type in absorbtions.keys():
        df[f'WOG_{soil_type}'] = 0.0
        
        # Apply cumulative calculation using expanding window
        for soil_type, rate in absorbtions.items():
            # Create temporary series for calculation
            wog = pd.Series(0.0, index=df.index)
            
            # Loop is necessary because each calculation depends on previous result
            for i in range(1, len(df)):
                wog.iloc[i] = max(0, wog.iloc[i-1] * (1 - rate) + df['Nedbor'].iloc[i])
                
            # Assign back to dataframe
            df[f'WOG_{soil_type}'] = wog

    df.dropna(inplace=True)

    df["is_rain"] = (df["WOG_clay"] > 5).astype(int) # IMPORTANT SHIAT!

    # Compute durations of dry spells
    dry_spells = []
    current_duration = 0
    censored = False  # True if the last dry spell was ongoing at the end of the dataset
    durations = []

    for idx, row in df.iterrows():
        if row["is_rain"] == 0:
            current_duration += 1
        else:
            if current_duration > 0:
                dry_spells.append({"duration": current_duration, "observed": 1})
                current_duration = 0
        durations.append(current_duration)
    # Handle censoring (if the dataset ends with a dry spell)
    if current_duration > 0:
        dry_spells.append({"duration": current_duration, "observed": 0})

    # Convert to DataFrame
    dry_spell_df = pd.DataFrame(dry_spells)

    df['duration'] = durations

    return df

def gather_soil_types(sediment_geo_json):
    # Gets all possible soil types and matches them with the available data.
    # TODO
    pass

def train_all_models(df, soiltypes):
    # Will train a model for eahc soiltype (depending on the availability of data for that soiltype.)
    survival_models = dict()
    # TODO







