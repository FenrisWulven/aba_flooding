
#=================================================================================================#
# THis script generates a summary of the statistics for a given parquet file containing precipitation data.
# It includes basic information, NaN statistics, earliest and latest records, value statistics, and data completeness over time.
# The script is designed to be run in a Python environment with the necessary libraries installed.
# It is assumed that the parquet file is already created and available in the specified path.
# The script uses the pandas library to read the parquet file and perform data analysis.
#=================================================================================================#




import pandas as pd
import numpy as np
from datetime import datetime

def generate_stats_summary(parquet_file):
    # Load the parquet file
    print(f"Loading data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    # Basic info
    print("\n==== BASIC INFORMATION ====")
    print(f"DataFrame shape: {df.shape}")
    print(f"Number of timestamps: {len(df)}")
    print(f"Number of stations: {len(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total data points: {df.size}")
    
    # NaN statistics
    print("\n==== NaN STATISTICS ====")
    total_nan = df.isna().sum().sum()
    total_values = df.size
    nan_percentage = (total_nan / total_values) * 100
    
    print(f"Total NaN values: {total_nan}")
    print(f"Percentage of NaN values: {nan_percentage:.2f}%")
    
    # NaN by station
    station_nan_counts = df.isna().sum()
    station_nan_percentage = (station_nan_counts / len(df)) * 100
    
    print("\nTop 5 stations with most NaNs:")
    for station, count in sorted(zip(station_nan_counts.index, station_nan_counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Station {station}: {count} NaNs ({station_nan_percentage[station]:.2f}%)")
    
    print("\nTop 5 stations with least NaNs:")
    for station, count in sorted(zip(station_nan_counts.index, station_nan_counts), key=lambda x: x[1])[:5]:
        print(f"  Station {station}: {count} NaNs ({station_nan_percentage[station]:.2f}%)")
    
    # Earliest non-NaN value for each station
    print("\n==== EARLIEST NON-NaN VALUES ====")
    
    earliest_values = {}
    for station in df.columns:
        # Find first non-NaN value
        non_nan_series = df[station].dropna()
        if len(non_nan_series) > 0:
            earliest_date = non_nan_series.index.min()
            earliest_value = non_nan_series.iloc[0]
            earliest_values[station] = (earliest_date, earliest_value)
    
    # Sort by earliest date
    sorted_earliest = sorted(earliest_values.items(), key=lambda x: x[1][0])
    
    print("First 5 stations with earliest records:")
    for i, (station, (date, value)) in enumerate(sorted_earliest[:5]):
        print(f"  {i+1}. Station {station}: First record on {date}, value: {value}")
    
    # Latest first appearance
    print("\nLast 5 stations to appear in the dataset:")
    for i, (station, (date, value)) in enumerate(sorted_earliest[-5:]):
        print(f"  {i+1}. Station {station}: First record on {date}, value: {value}")
    
    # Value statistics
    print("\n==== VALUE STATISTICS ====")
    
    # Remove NaN values for statistics
    non_nan_values = df.values[~np.isnan(df.values)]
    
    print(f"Minimum value: {np.min(non_nan_values)}")
    print(f"Maximum value: {np.max(non_nan_values)}")
    print(f"Mean value: {np.mean(non_nan_values):.4f}")
    print(f"Median value: {np.median(non_nan_values):.4f}")
    print(f"Standard deviation: {np.std(non_nan_values):.4f}")
    
    # Count zero values
    zero_count = (df == 0).sum().sum()
    zero_percentage = (zero_count / (total_values - total_nan)) * 100
    print(f"Zero values: {zero_count} ({zero_percentage:.2f}% of non-NaN values)")
    
    # Count values above certain thresholds
    thresholds = [1, 5, 10, 20, 50]
    for threshold in thresholds:
        count = (df > threshold).sum().sum()
        percentage = (count / (total_values - total_nan)) * 100
        print(f"Values > {threshold}: {count} ({percentage:.4f}% of non-NaN values)")
    
    # Data completeness over time - FIXED
    print("\n==== DATA COMPLETENESS OVER TIME ====")
    
    # Create a temporary copy to avoid modifying the original
    temp_df = df.copy()
    
    # Group by year
    yearly_groups = temp_df.groupby(temp_df.index.year)
    
    # Calculate completeness correctly
    yearly_stats = []
    for year, group in yearly_groups:
        # Total possible data points for this year
        total_possible = group.shape[0] * group.shape[1]  # rows * columns
        
        # Actual non-NaN values
        actual_values = total_possible - group.isna().sum().sum()
        
        # Percentage completeness
        completeness = (actual_values / total_possible) * 100
        
        yearly_stats.append((year, completeness))
    
    # Sort by completeness
    yearly_stats.sort(key=lambda x: x[1], reverse=True)
    
    print("Data completeness by year (top 5):")
    for year, pct in yearly_stats[:5]:
        print(f"  {year}: {pct:.2f}% complete")
    
    print("\nData completeness by year (bottom 5):")
    for year, pct in sorted(yearly_stats, key=lambda x: x[1])[:5]:
        print(f"  {year}: {pct:.2f}% complete")
    
    print("\n==== SUMMARY COMPLETE ====")

if __name__ == "__main__":
    parquet_file = "precipitation_data.parquet"
    generate_stats_summary(parquet_file)