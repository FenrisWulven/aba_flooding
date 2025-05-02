import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Helper function to scan through station files
def scan_data_features(processed_data_path="data/processed/"):
    """Scan through station files to identify available features"""
    
    all_features = set()
    column_counts = {}
    
    # List all station files
    station_files = [f for f in os.listdir(processed_data_path) 
                    if f.startswith("survival_data_05005") and f.endswith(".parquet")]
    
    print(f"Found {len(station_files)} station files")
    
    # Sample from first file for feature analysis
    if station_files:
        sample_file = os.path.join(processed_data_path, station_files[0])
        sample_df = pd.read_parquet(sample_file)
        
        print("\nSample data from first station file:")
        print(f"Station ID: {station_files[0].replace('survival_data_','').replace('.parquet','')}")
        print(f"Data shape: {sample_df.shape}")
        
        # Show feature patterns
        print("\nAvailable feature patterns:")
        for col in sample_df.columns:
            parts = col.split('_')
            if len(parts) >= 3:
                pattern = f"*_{parts[1]}_{parts[2:]}"
                all_features.add(pattern)
                column_counts[pattern] = column_counts.get(pattern, 0) + 1
        
        # Display sample data for model training
        print("\nExample data for DeepCox model training:")
        
        # Get a station and soil type from the sample
        station = sample_df.columns[0].split('_')[0]
        soil_types = set()
        
        for col in sample_df.columns:
            parts = col.split('_')
            if len(parts) >= 3 and parts[0] == station and parts[1] != "WOG":
                soil_types.add(parts[1])
        
        if soil_types:
            soil_type = list(soil_types)[0]
            print(f"\nSample for station {station}, soil type {soil_type}:")
            
            # Column names
            duration_col = f"{station}_{soil_type}_duration"
            event_col = f"{station}_{soil_type}_observed" 
            tte_col = f"{station}_{soil_type}_TTE"
            wog_col = f"{station}_WOG_{soil_type}"
            
            cols_to_show = [col for col in [duration_col, event_col, tte_col, wog_col] 
                           if col in sample_df.columns]
            
            if cols_to_show:
                display_df = sample_df[cols_to_show].head(10)
                print(display_df)
                
                # Show statistics
                print("\nFeature statistics:")
                stats_df = sample_df[cols_to_show].describe()
                print(stats_df)
    
    return all_features, column_counts

def plot_tte_over_time(station_id, processed_data_path="data/processed/", max_soil_types=1, y_limit=30000):
    """
    Plot Time To Event (TTE) values over time for soil types in a station
    
    Parameters:
    -----------
    station_id : str
        Station identifier
    processed_data_path : str
        Path to processed data files
    max_soil_types : int
        Maximum number of soil types to include in the plot
    y_limit : int
        Maximum y-axis value to display
    """
    
    # Construct file path for the station
    file_path = os.path.join(processed_data_path, f"survival_data_{station_id}.parquet")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the data
    df = pd.read_parquet(file_path)
    print(f"Loaded data for station {station_id}: {df.shape}")
    
    # Find all TTE columns for this station
    tte_columns = [col for col in df.columns if f"{station_id}_" in col and "_TTE" in col]
    
    if not tte_columns:
        print(f"No TTE columns found for station {station_id}")
        return
        
    # Limit to the specified maximum number of soil types
    if len(tte_columns) > max_soil_types:
        print(f"Limiting plot to {max_soil_types} soil types out of {len(tte_columns)} available")
        tte_columns = tte_columns[:max_soil_types]
    
    # Add a datetime index for plotting if it doesn't exist
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to create a date range index for better visualization
        try:
            # Assuming hourly data with no gaps
            # calculate start time based on today and then minus the number of hours (rows) 
            start_time = pd.Timestamp.now() - pd.Timedelta(hours=len(df))
            df.index = pd.date_range(start=start_time, periods=len(df), freq='H')
            print("Created synthetic datetime index for plotting")
        except Exception as e:
            print(f"Could not create datetime index: {e}")
    
    # Plot TTE values for all soil types
    plt.figure(figsize=(12, 8))
    
    for col in tte_columns:
        # Extract soil type from column name
        soil_type = col.split('_')[1]
        # Plot TTE values
        plt.plot(df.index, df[col], label=f"Soil: {soil_type}")
    
    plt.title(f"Time To Event (TTE) Over Time for Station {station_id}")
    plt.xlabel("Date/Time")
    plt.ylabel("Time To Next Event (Hours)")
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limit
    plt.ylim(0, y_limit)
    
    # Add markers for events (where TTE = 0)
    for col in tte_columns:
        event_points = df[df[col] == 0]
        if not event_points.empty:
            plt.scatter(event_points.index, event_points[col], 
                      marker='o', s=10, color='red', zorder=5, 
                      label=f"Flood Events" if col == tte_columns[0] else "")
    
    # Save the plot
    output_path = f"./outputs/plots/tte_time_series_{station_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved TTE time series plot to {output_path}")
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Run the scanning function
if __name__ == "__main__":
    features, counts = scan_data_features()
    
    print("\n=== RECOMMENDATION FOR DEEP COX FEATURES ===")
    print("Based on the data structure, you should use these patterns in deep_cox_features:")
    print("""
deep_cox_features = [
    "WOG",      # Water on ground measurements 
    "TTE",      # Time to event information
    "duration"  # Duration since last event
]
    """)
    
    print("For specific station-soil combinations, you can use:")
    print("""
# For all stations but specific soil type (e.g., DS):
deep_cox_features = [col for col in df.columns if "_WOG_DS" in col or "_DS_TTE" in col]

# For a specific station and all its features:
station_id = "06058"
deep_cox_features = [col for col in df.columns if col.startswith(f"{station_id}_")]
    """)
    
    # Plot TTE over time for a sample station
    # Change this station ID to any station you want to analyze
    station_to_analyze = "05085"  # Example station
    plot_tte_over_time(station_to_analyze)

