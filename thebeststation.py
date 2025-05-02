import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Set your parquet file path here
PARQUET_FILE_PATH = "data/raw/precipitation_data.parquet"

def analyze_rainfall(parquet_file):
    """
    Analyze rainfall data across multiple stations from a parquet file.
    Identifies which station had the most rain in a single day and the highest total rainfall.
    
    Parameters:
    -----------
    parquet_file : str
        Path to the parquet file containing rainfall data
    
    Returns:
    --------
    tuple
        (max_single_day_station, max_single_day_value, max_single_day_date,
         max_total_station, max_total_value)
    """
    try:
        # Read the parquet file
        print(f"Reading rainfall data from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        # Print basic information about the dataset
        print(f"\nDataset information:")
        print(f"Time period: {df.index.min()} to {df.index.max()}")
        print(f"Number of days: {len(df)}")
        print(f"Number of weather stations: {len(df.columns)}")
        print(f"Station IDs: {', '.join(df.columns)}")
        
        # Check for duplicate dates
        print("\nChecking for duplicate dates...")
        if df.index.duplicated().any():
            dup_dates = df.index[df.index.duplicated(keep=False)].unique()
            print(f"WARNING: Found {len(dup_dates)} duplicate dates:")
            for date in dup_dates:
                count = df.index.to_series().value_counts()[date]
                print(f"  - {date}: appears {count} times")
            print("Consider aggregating or resolving duplicate entries before analysis.")
        else:
            print("No duplicate dates found. Each date appears exactly once.")
        
        # Find TOP 20 stations with maximum single-day rainfall
        max_rainfall = df.max().sort_values(ascending=False)
        top20_single_day = max_rainfall.head(20)
        
        # Find the dates of maximum rainfall for TOP 20 stations
        top20_dates = {}
        for station in top20_single_day.index:
            top20_dates[station] = df[station].idxmax()
        
        # Find TOP 20 stations with maximum total rainfall
        total_rainfall = df.sum().sort_values(ascending=False)
        top20_total = total_rainfall.head(20)
        
        # Print results
        print("\n===== RESULTS =====")
        print("\nTOP 20 STATIONS - MOST RAIN IN A SINGLE DAY:")
        for i, (station, value) in enumerate(top20_single_day.items(), 1):
            print(f"{i}. {station}: {value:.2f} mm on {top20_dates[station]}")
        
        print("\nTOP 20 STATIONS - HIGHEST TOTAL RAINFALL:")
        for i, (station, value) in enumerate(top20_total.items(), 1):
            print(f"{i}. {station}: {value:.2f} mm")
        
        # Return the results (keeping the original return values for compatibility)
        max_station_single_day = top20_single_day.index[0]
        max_value_single_day = top20_single_day.iloc[0]
        max_date = top20_dates[max_station_single_day]
        max_station_total = top20_total.index[0]
        max_value_total = top20_total.iloc[0]
        
        return (max_station_single_day, max_value_single_day, max_date, 
                max_station_total, max_value_total)
        
    except Exception as e:
        print(f"Error analyzing rainfall data: {e}")
        return None

def main():
    # Use the global variable defined at the top of the script
    parquet_file = PARQUET_FILE_PATH
    print(f"Using parquet file: {parquet_file}")
    
    analyze_rainfall(parquet_file)

if __name__ == "__main__":
    main()