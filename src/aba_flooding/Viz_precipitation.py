import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import seaborn as sns
from datetime import datetime
import os

# Function to load and preprocess the data
def load_data(file_path):
    # Read parquet file
    data = pq.read_table(file_path).to_pandas()
    
    # Reset index if it's not a proper datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        # Assuming there's a datetime column, otherwise you'll need to create one
        if 'datetime' in data.columns:
            data.set_index('datetime', inplace=True)
        else:
            # Create a datetime index using the row number and assuming hourly data
            # Starting from a specific date (modify as needed)
            start_date = datetime(1990, 1, 1)
            hours = pd.date_range(start=start_date, periods=len(data), freq='H')
            data.index = hours
    
    return data

# Function to visualize data completeness for all stations
def plot_data_completeness(data, save_path=None):
    # Get all stations
    all_stations = data.columns.tolist()
    
    # Calculate percentage of NaN values for each station
    nan_percentage = data[all_stations].isna().mean() * 100
    
    # Sort by NaN percentage for better visualization
    nan_percentage = nan_percentage.sort_values()
    
    plt.figure(figsize=(15, 10))
    plt.bar(nan_percentage.index, nan_percentage.values)
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Percentage of Missing Data (NaN) by Station')
    plt.ylabel('NaN Percentage (%)')
    plt.xlabel('Weather Station')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/data_completeness_all.png", dpi=300)
    else:
        plt.show()
    plt.close()

# Function to plot seasonal patterns (monthly averages) for top stations
def plot_seasonal_patterns(data, top_stations, save_path=None):
    # Extract month from datetime index
    data_monthly = data.copy()
    data_monthly['month'] = data_monthly.index.month
    
    # Calculate monthly averages for top stations
    monthly_avg = data_monthly.groupby('month')[top_stations].mean()
    
    # Plot each station's seasonal pattern
    plt.figure(figsize=(12, 8))
    
    for station in top_stations:
        plt.plot(monthly_avg.index, monthly_avg[station], label=station)
    
    plt.title('Monthly Average Precipitation by Station')
    plt.xlabel('Month')
    plt.ylabel('Average Precipitation')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/seasonal_patterns.png", dpi=300)
    else:
        plt.show()
    plt.close()

# Function to visualize annual precipitation totals for each station
def plot_station_comparison(data, top_stations, save_path=None):
    # Calculate total precipitation for each station
    total_precip = data[top_stations].sum()
    
    plt.figure(figsize=(12, 8))
    plt.bar(total_precip.index, total_precip.values)
    plt.title('Total Precipitation by Station')
    plt.ylabel('Total Precipitation')
    plt.xlabel('Weather Station')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/station_comparison.png", dpi=300)
    else:
        plt.show()
    plt.close()

# Function to plot heatmap of data availability with years 5 years apart
# Function to plot heatmap of data availability with years 5 years apart
def plot_data_availability_heatmap(data, save_path=None):
    # Get all stations
    all_stations = data.columns.tolist()
    
    # Create a binary mask: 1 for data present, 0 for NaN
    data_mask = ~data[all_stations].isna()
    
    # Resample to monthly data for better visualization
    monthly_data = data_mask.resample('ME').sum()
    
    # Create year labels for x-axis (5 years apart)
    years = monthly_data.index.year.unique()
    start_year = years[0]
    end_year = years[-1]
    
    # Create a list of years 5 years apart
    selected_years = list(range(start_year, end_year + 1, 5))
    
    # Create a list of positions for these years in the x-axis
    year_positions = []
    year_labels = []
    
    for year in selected_years:
        # Find the position of January for each selected year
        positions = [i for i, date in enumerate(monthly_data.index) 
                    if date.year == year and date.month == 1]
        
        if positions:
            year_positions.append(positions[0])
            year_labels.append(str(year))
    
    # Calculate figure height based on number of stations
    # Each station should have at least 0.2 inches in height
    height = max(15, len(all_stations) * 0.2)
    
    # Create a figure big enough to show all stations
    plt.figure(figsize=(20, height))
    
    # Sort the stations by completeness for better visualization
    completeness = data_mask.mean().sort_values(ascending=False)
    sorted_stations = completeness.index.tolist()
    
    # Use the sorted stations for the heatmap
    sns.heatmap(monthly_data[sorted_stations].T, cmap='viridis', 
                xticklabels=False,  # We'll set custom x-ticks
                yticklabels=True,   # Show all station labels
                cbar_kws={'label': 'Hours with data'})
    
    # Set custom x-ticks with years 5 years apart
    plt.xticks(year_positions, year_labels)
    
    # Make y-ticks (station labels) smaller if there are many
    if len(all_stations) > 30:
        plt.yticks(fontsize=8)
    
    plt.title('Monthly Data Availability (Hours with Data)')
    plt.ylabel('Weather Station')
    plt.xlabel('Year')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/data_availability_heatmap_all.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# Function to plot precipitation intensity distribution
def plot_precipitation_distribution(data, top_stations, save_path=None):
    plt.figure(figsize=(12, 8))
    
    for station in top_stations:
        # Filter out NaN values
        station_data = data[station].dropna()
        # Filter only precipitation events (> 0)
        station_data = station_data[station_data > 0]
        
        if len(station_data) > 0:  # Check if there's data to plot
            sns.kdeplot(station_data, label=station)
    
    plt.title('Distribution of Precipitation Intensity')
    plt.xlabel('Precipitation Amount')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/precipitation_distribution.png", dpi=300)
    else:
        plt.show()
    plt.close()

# Function to plot a geographical heatmap of total precipitation
def plot_geographical_heatmap(data, station_coordinates, save_path=None):
    """
    This function requires a dictionary of station coordinates 
    in the format {station_id: (latitude, longitude)}
    """
    # Calculate total precipitation for each station
    total_precip = data.sum()
    
    # Create lists for scatter plot
    stations = []
    latitudes = []
    longitudes = []
    precipitations = []
    
    for station in total_precip.index:
        if station in station_coordinates:
            stations.append(station)
            lat, lon = station_coordinates[station]
            latitudes.append(lat)
            longitudes.append(lon)
            precipitations.append(total_precip[station])
    
    plt.figure(figsize=(12, 10))
    plt.scatter(longitudes, latitudes, c=precipitations, cmap='Blues', 
                s=100, alpha=0.7, edgecolors='k')
    
    # Add station labels
    for i, station in enumerate(stations):
        plt.annotate(station, (longitudes[i], latitudes[i]), 
                    fontsize=8, ha='center', va='bottom')
    
    plt.colorbar(label='Total Precipitation')
    plt.title('Geographical Distribution of Total Precipitation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/geographical_heatmap.png", dpi=300)
    else:
        plt.show()
    plt.close()

# Main function to run the analysis
def main(file_path, output_dir=None):
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the data
    print("Loading data...")
    data = load_data(file_path)
    
    # Get all station IDs
    all_stations = data.columns.tolist()
    print(f"Total number of stations: {len(all_stations)}")
    
    # Get top 10 stations by data availability for the detailed plots
    non_nan_counts = data.count()
    top_10_stations = non_nan_counts.sort_values(ascending=False).head(10).index.tolist()
    print(f"Top 10 stations selected for detailed analysis: {top_10_stations}")
    
    # Plot data completeness for all stations
    print("Plotting data completeness for all stations...")
    plot_data_completeness(data, output_dir)
    
    # Plot seasonal patterns for top 10 stations
    print("Plotting seasonal patterns for top 10 stations...")
    plot_seasonal_patterns(data, top_10_stations, output_dir)
    
    # Plot station comparison for top 10 stations
    print("Plotting station comparison for top 10 stations...")
    plot_station_comparison(data, top_10_stations, output_dir)
    
    # Plot data availability heatmap for all stations
    print("Plotting data availability heatmap for all stations...")
    plot_data_availability_heatmap(data, output_dir)
    
    # Plot precipitation distribution for top 10 stations
    print("Plotting precipitation distribution for top 10 stations...")
    plot_precipitation_distribution(data, top_10_stations, output_dir)
    
    # Note: Uncomment this if you have station coordinates
    # Example of how you would create the station_coordinates dictionary
    # station_coordinates = {
    #     '06019': (55.7, 12.3),  # Replace with actual coordinates
    #     '06109': (55.6, 12.4),
    #     # Add coordinates for all stations
    # }
    # plot_geographical_heatmap(data, station_coordinates, output_dir)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "data/raw/precipitation_imputed_data.parquet"
    
    # Optional: provide an output directory to save the plots
    output_dir = "precipitation_visualizations_imputed"
    
    main(file_path, output_dir)