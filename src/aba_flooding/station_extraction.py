

#=======================================================================================================#
# This script extracts station information from the DMI API and saves it to a Parquet file. It also     #
# creates a map visualization of the stations using Folium. The script first attempts to access the     #
# stations endpoint directly. If that fails, it falls back to extracting unique station IDs from        #
# observation data. The script handles potential issues with duplicate columns and missing coordinates. #
# It also includes error handling for API requests and file saving. The map visualization is saved as   #
# an HTML file in the specified directory. The script is designed to be run in a Python environment##   #
# with the necessary libraries installed.                                                               #
#=======================================================================================================#

import requests
import pandas as pd
import folium
import os

# Replace this with your actual API key
api_key = 'd111ba1d-a1f5-43a5-98c6-347e9c2729b2'  # insert your own key here

# Method 1: Try to access stations endpoint directly
stations_url = 'https://dmigw.govcloud.dk/v2/metObs/collections/station/items'

def get_all_stations_method1():
    """
    Attempt to get all stations using a dedicated station endpoint
    """
    try:
        response = requests.get(stations_url, params={'api-key': api_key})
        if response.status_code == 200:
            stations_data = response.json()
            
            # Print the first feature to debug the structure
            if 'features' in stations_data and stations_data['features']:
                print("First station feature structure:")
                print(stations_data['features'][0])
            
            # Create a fresh DataFrame to avoid duplicates
            stations_list = []
            for feature in stations_data['features']:
                station = {}
                
                # Extract basic info
                station['id'] = feature.get('id')
                station['feature_type'] = feature.get('type')
                
                # Extract geometry
                if 'geometry' in feature:
                    if 'coordinates' in feature['geometry']:
                        coords = feature['geometry']['coordinates']
                        if coords and len(coords) >= 2:
                            station['longitude'] = coords[0]
                            station['latitude'] = coords[1]
                    if 'type' in feature['geometry']:
                        station['geometry_type'] = feature['geometry']['type']
                
                # Extract properties
                if 'properties' in feature:
                    props = feature['properties']
                    for key, value in props.items():
                        # Rename 'type' to avoid duplicates
                        if key == 'type':
                            station['station_type'] = value
                        else:
                            station[key] = value
                
                stations_list.append(station)
            
            # Create DataFrame from our clean list
            stations_df = pd.DataFrame(stations_list)
            
            return stations_df
        else:
            print(f"Failed to access station endpoint. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error accessing station endpoint: {e}")
        return None

# Method 2: Extract unique stations from observation data
def get_all_stations_method2():
    """
    Get all stations by extracting unique station IDs from observation data
    """
    dmi_url = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items'
    
    try:
        # Request with a high limit to get as many records as possible
        params = {
            'api-key': api_key,
            'limit': '300000'  # Maximum allowed limit
        }
        
        response = requests.get(dmi_url, params=params)
        if response.status_code != 200:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None
            
        json_data = response.json()
        
        # Create a fresh DataFrame to avoid duplicates
        stations_list = []
        seen_station_ids = set()
        
        # Process each feature
        for feature in json_data.get('features', []):
            # Only process if it has properties and stationId
            if 'properties' in feature and 'stationId' in feature['properties']:
                station_id = feature['properties']['stationId']
                
                # Skip if we've already seen this station
                if station_id in seen_station_ids:
                    continue
                
                seen_station_ids.add(station_id)
                
                station = {'stationId': station_id}
                
                # Extract coordinates if available
                if 'geometry' in feature and 'coordinates' in feature['geometry']:
                    coords = feature['geometry']['coordinates']
                    if coords and len(coords) >= 2:
                        station['longitude'] = coords[0]
                        station['latitude'] = coords[1]
                
                stations_list.append(station)
        
        # Create DataFrame from our clean list
        stations_df = pd.DataFrame(stations_list)
            
        return stations_df
    
    except Exception as e:
        print(f"Error extracting stations from observation data: {e}")
        return None

# Try both methods and use the one that works
print("Attempting to get stations using Method 1...")
stations_df = get_all_stations_method1()
if stations_df is None or stations_df.empty:
    print("Falling back to method 2...")
    stations_df = get_all_stations_method2()

if stations_df is not None and not stations_df.empty:
    print(f"Successfully retrieved {len(stations_df)} stations")
    print("\nColumns in the DataFrame:")
    print(stations_df.columns.tolist())
    print("\nFirst 10 stations:")
    print(stations_df.head(10))
    
    # Check for duplicate column names
    if len(stations_df.columns) != len(set(stations_df.columns)):
        duplicate_cols = [col for col in stations_df.columns if list(stations_df.columns).count(col) > 1]
        print(f"Warning: Found duplicate columns: {duplicate_cols}")
    
    # Save to Parquet in the specified directory
    output_dir = '/Users/maks/Documents/GitHub/aba_flooding/dmi_data_daily'
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'dmi_stations.parquet')
    
    try:
        # Save to parquet
        stations_df.to_parquet(output_path, index=False)
        print(f"\nSaved station information to '{output_path}'")
        
    except Exception as e:
        print(f"Error saving to Parquet: {e}")
        print("Detailed column information for debugging:")
        for i, col in enumerate(stations_df.columns):
            print(f"{i}: {col} - type: {stations_df[col].dtype}")
        
        print("\nPlease fix the column issues and try again.")
    
    # Create a map visualization if coordinates are available
    if 'longitude' in stations_df.columns and 'latitude' in stations_df.columns:
        # Filter out rows with missing coordinates
        map_df = stations_df.dropna(subset=['longitude', 'latitude'])
        
        if len(map_df) > 0:
            # Calculate the center of the map
            center_lat = map_df['latitude'].mean()
            center_lon = map_df['longitude'].mean()
            
            # Create a map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
            
            # Add markers for each station
            for _, row in map_df.iterrows():
                # Create popup with station info
                popup_html = f"""
                <b>Station ID:</b> {row.get('stationId', 'N/A')}<br>
                """
                
                # Add name if available
                if 'name' in row and pd.notna(row['name']):
                    popup_html += f"<b>Name:</b> {row['name']}<br>"
                
                # Add station type if available
                if 'station_type' in row and pd.notna(row['station_type']):
                    popup_html += f"<b>Type:</b> {row['station_type']}<br>"
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color='blue')
                ).add_to(m)
            
            # Save the map to the same directory
            map_path = os.path.join(output_dir, 'dmi_stations_map.html')
            m.save(map_path)
            print(f"Created map visualization in '{map_path}'")
else:
    print("Failed to retrieve station information using both methods.")