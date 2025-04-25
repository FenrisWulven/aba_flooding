import pandas as pd

# Execute the function to create and display the Voronoi map
if __name__ == "__main__":
    df = pd.read_parquet('data/raw/precipitation_imputed_data.parquet')
    df = pd.read_parquet('data/raw/dmi_stations.parquet')
    print(df.columns)
    print(df.head())

