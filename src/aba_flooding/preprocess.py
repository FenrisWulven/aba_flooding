import pandas as pd
import aba_flooding.perculation_mapping as pm
import aba_flooding.geo_utils as gu
# import perculation_mapping as pm
# import geo_utils as gu
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
import os
import matplotlib.pyplot as plt
###########################################
# SECTION 1: GEOGRAPHIC DATA PROCESSING   #
###########################################

def voronoi_finite_polygons_2d(vor, radius=None):
    """Convert Voronoi diagram to finite polygons."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    
    new_regions = []
    new_vertices = vor.vertices.tolist()
    
    center = vor.points.mean(axis=0)
    radius = np.ptp(vor.points, axis=0).max() * 2 if radius is None else radius
    
    # Construct a map of all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        # Skip points that don't have any ridges
        if p1 not in all_ridges:
            print(f"Skipping point {p1} which has no ridges")
            continue
            
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue
        
        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge
                continue
            
            # Infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices.mean(axis=0) + direction * radius
            
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        
        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        
        new_regions.append(new_region.tolist())
    
    return new_regions, np.asarray(new_vertices)

def create_precipitation_coverage(denmark_gdf):
    """
    Create Voronoi polygons for precipitation stations that cover Denmark without overlap.
    
    Called by: create_full_coverage()
    Calls: voronoi_finite_polygons_2d()
    """
    try:
        # Load dmi station data - stations are rows with longitude and latitude columns
        print(f"Loading dmi station data from data/raw/dmi_stations.parquet...")
        station_data = pd.read_parquet('data/raw/dmi_stations.parquet')
        print(f"Loaded coordinate data of the station with {len(station_data)} stations")
        
        # Determine the station ID column
        if 'stationId' in station_data.columns:
            id_column = 'stationId' 
        else:
            print("WARNING: No 'stationId' column found, using first column as ID")   
            id_column = station_data.columns[0]  # Use the first column as ID 
        
        # Find longitude and latitude columns
        lon_col = next((col for col in station_data.columns if col.lower() in ['longitude', 'lon', 'long']), None)
        lat_col = next((col for col in station_data.columns if col.lower() in ['latitude', 'lat']), None)
        
        if lon_col is None or lat_col is None:
            raise ValueError(f"Could not identify longitude and latitude columns. Available columns: {station_data.columns.tolist()}")
        print(f"Using column '{lon_col}' for longitude, '{lat_col}' for latitude, and '{id_column}' for station IDs")
        
        # Create GeoDataFrame for stations
        print("Creating GeoDataFrame for stations containing their coordinates")
        stations_gdf = gpd.GeoDataFrame(
            station_data,
            geometry=gpd.points_from_xy(station_data[lon_col], station_data[lat_col]),
            crs="EPSG:4326"  # WGS84 ellipsoid is a coordinate system used in Google Earth and GSP systems
        )
        
        # Reproject to match map projection since the Voronoi diagram is in EPSG:3857
        print("Reprojecting to Web Mercator EPSG:3857 for Voronoi diagram calculations")
        stations_gdf = stations_gdf.to_crs("EPSG:3857")  # Web Mercator
        # This EPSG:3857 makes the the X/Y coordinates in meters, which is suitable for Voronoi diagram calculations
        # Basically it makes it square, so the Voronoi diagram is not distorted by the curvature of the earth
        
        # Print information about Denmark boundary
        # print(f"Denmark GDF info: {denmark_gdf.shape}")
        # print(f"Denmark GDF columns: {denmark_gdf.columns.tolist()}")
        # print(f"Denmark CRS: {denmark_gdf.crs}")
        
        # Create Voronoi diagram
        print("Creating Voronoi diagram...")
        coords = np.array([(p.x, p.y) for p in stations_gdf.geometry])
        print(f"Number of station coordinates: {len(coords)}")
        
        # Check for duplicate or very close points
        _, unique_indices = np.unique(np.round(coords, decimals=5), axis=0, return_index=True)
        if len(unique_indices) < len(coords):
            print(f"WARNING: Found {len(coords) - len(unique_indices)} potential duplicate stations. Using only unique locations.")
            coords = coords[np.sort(unique_indices)]
            # Adjust stations_gdf to match unique points
            stations_gdf = stations_gdf.iloc[np.sort(unique_indices)].copy()
        
        # Get Denmark boundary for clipping
        boundary = denmark_gdf.geometry.union_all().bounds
        print(f"Denmark bounds:")
        print(f"\tSW corner city: {boundary[0]}, {boundary[1]}")
        print(f"\tNE corner city: {boundary[2]}, {boundary[3]}")

        boundary_width = boundary[2] - boundary[0]
        boundary_height = boundary[3] - boundary[1]

        # Add corner points to ensure complete coverage
        corner_points = [
            [boundary[0] - boundary_width, boundary[1] - boundary_height],
            [boundary[2] + boundary_width, boundary[1] - boundary_height],
            [boundary[0] - boundary_width, boundary[3] + boundary_height],
            [boundary[2] + boundary_width, boundary[3] + boundary_height]
        ]
        
        all_points = np.vstack([coords, corner_points])
        print(f"Total points for Voronoi (including corners): {len(all_points)}")
        
        try:
            vor = Voronoi(all_points)
            print(f"Voronoi diagram created with {len(vor.points)} points and {len(vor.vertices)} vertices")
        except Exception as vor_error:
            print(f"ERROR creating Voronoi diagram: {vor_error}")
            # Add jitter to points to avoid collinearity issues
            jitter = np.random.normal(0, 0.00001, all_points.shape)
            all_points = all_points + jitter
            print("Added small jitter to points to avoid numerical issues, retrying...")
            vor = Voronoi(all_points)
        
        # Get Voronoi polygons
        print("Converting Voronoi diagram to polygons...")
        regions, vertices = voronoi_finite_polygons_2d(vor)
        print(f"Created {len(regions)} Voronoi regions")
        
        # Create clipped polygons for each station 
        # this is because the Voronoi polygons can be infinite. So we need to clip them to the Denmark boundary
        print("Creating clipped polygons so that they are within the Denmark boundary...")
        voronoi_polygons = []
        valid_station_ids = []
        
        for i, region in enumerate(regions):
            if i < len(coords):  # Skip corner points
                try:
                    polygon = Polygon([vertices[v] for v in region])
                    # Clip polygon to Denmark boundary
                    clipped_polygon = polygon.intersection(denmark_gdf.geometry.union_all())
                    if not clipped_polygon.is_empty:
                        voronoi_polygons.append(clipped_polygon)
                        valid_station_ids.append(stations_gdf.iloc[i][id_column])
                except Exception as poly_error:
                    print(f"ERROR creating polygon for region {i}: {poly_error}")
                    continue
        
        print(f"Created {len(voronoi_polygons)} valid polygons")
        
        # Create GeoDataFrame with coverage areas, meaning that the polygons are the coverage areas of the stations
        coverage_gdf = gpd.GeoDataFrame(
            {'station_id': valid_station_ids},
            geometry=voronoi_polygons,
            crs=stations_gdf.crs
        )
        
        # TODO: Add avg_precipitation data to coverage areas if available
        print("Adding avg preciptation to the station data of the polygons...")
        # load in the precipitation data
        precipitation_data = pd.read_parquet('data/raw/precipitation_imputed_data.parquet')
        precipitation_data = precipitation_data.clip(lower=0, upper=100) 
        # drop nans
        # precipitation_data.dropna(inplace=True) # inplace means that 

        # precipitation_data has columns indexed by station IDs with the mm values
        avg_prec = precipitation_data.mean(axis=0, skipna=True)
        print(f"Highest average precipitation: {avg_prec.max()}")
        print(f"Lowest average precipitation: {avg_prec.min()}")

        # check the avg_precipitation values as a histogram
        plt.figure(figsize=(10, 6))
        plt.title("Average Hourly Precipitation Histogram")
        plt.xlabel("Average Hourly Precipitation (mm)")
        plt.ylabel("Frequency")
        plt.hist(avg_prec, bins=30, color='blue', alpha=0.7)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig("outputs/plots/avg_precipitation_histogram.png")
        
        # Return the coverage GeoDataFrame and stations GeoDataFrame
        # File saving is handled in create_full_coverage() to avoid duplication
        return coverage_gdf, stations_gdf
    
    except Exception as e:
        print(f"ERROR creating precipitation coverage: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_full_coverage():
    """
    Create coverage areas for precipitation stations across Denmark.
    
    Returns:
    --------
    tuple: (GeoJSONDataSource, GeoDataFrame, GeoDataFrame)
        Coverage as GeoJSON source - contains geo data in GeoJSON format
        coverage GeoDataFrame      - 
        stations GeoDataFrame
    
    Called by: main
    Calls: create_precipitation_coverage(), gu.gdf_to_geojson()
    """
    # Create directories for output files
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # Create a simplified Denmark boundary manually
    print("Creating simplified Denmark boundary...")
    # Approximate Denmark bounding box in EPSG:4326 (WGS84)
    # These coordinates represent a rough bounding box around Denmark
    denmark_coords = [
        (8.0, 54.5),   # Southwest 
        (8.0, 57.8),   # Northwest 
        (13.0, 57.8),  # Northeast 
        (13.0, 54.5),  # Southeast
        (8.0, 54.5)    # to close the polygon
    ]
    
    # Create a polygon and convert to GeoDataFrame
    denmark_polygon = Polygon(denmark_coords)
    denmark_polygon_gdf = gpd.GeoDataFrame(
        {'name': ['Denmark']}, 
        geometry=[denmark_polygon], 
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    print("Using simplified Denmark boundary")

    # Create station coverage areas
    print("\nCreating station coverage areas")
    coverage_geojson_gdf, stations_gdf = create_precipitation_coverage(denmark_polygon_gdf)
    
    if coverage_geojson_gdf is not None and not coverage_geojson_gdf.empty:
        print(f"Successfully created polygon coverage GeoDataFrame with {len(coverage_geojson_gdf)} polygons")
        
        # Save the GeoJSON file using various methods as fallbacks
        try:
            coverage_geojson_gdf.to_file("data/raw/precipitation_coverage.geojson", driver="GeoJSON")
            print("Saved coverage GeoDataFrame to data/raw/precipitation_coverage.geojson")
        except AttributeError as e:
            if "module 'pyogrio' has no attribute 'write_dataframe'" in str(e):
                print("ERROR saving 'precipitation_coverage.geojson' due to pyogrio error has no attribute 'write_dataframe'") 
                try:
                    # Try using fiona driver directly
                    import fiona
                    coverage_geojson_gdf.to_file(
                        "data/raw/precipitation_coverage.geojson", 
                        driver="GeoJSON",
                        engine="fiona"
                    )
                    print("Saved coverage GeoDataFrame using fiona engine")
                except Exception as fiona_error:
                    print(f"ERROR Fiona method failed also: {fiona_error}")
                    try:
                        # Last resort: manually create GeoJSON
                        import json
                        geojson_dict = json.loads(gu.gdf_to_geojson(coverage_geojson_gdf))
                        with open("data/raw/precipitation_coverage.geojson", "w") as f:
                            json.dump(geojson_dict, f)
                        print("Saved coverage GeoDataFrame using manual JSON conversion")
                    except Exception as json_error:
                        print(f"ERROR Manual JSON conversion failed: {json_error}")
                        print("ERROR: Could not save precipitation coverage file")
            else:
                print(f"ERROR Could not save precipitation coverage areas: {e}")
        except Exception as general_error:
            print(f"ERROR Could not save precipitation coverage areas: {general_error}")
        
        return coverage_geojson_gdf, stations_gdf
    
    else:
        print("No valid coverage areas created. Skipping GeoJSON creation.")
        return None, None, None

###########################################
# SECTION 2: SOIL AND SEDIMENT ANALYSIS   #
###########################################

def sediment_types_for_station(stationId, precipitationCoverageStations, sedimentCoverage):
    """
    Get all soil types contained within a station area.
    
    Parameters:
    -----------    
    stationId : str
        Station ID to filter soil types
    precipitationCoverageStations : geoDataFrame
        GeoDataFrame containing precipitation coverage data
    sedimentCoverage : geoDataFrame
        GeoDataFrame containing sediment coverage data
    
    Returns:
    --------
    list : List of soil types within the station area or empty list if station not found
    
    Called by: load_process_data()
    """
    # Debug information about the datasets
    print(f"Precipitation coverage CRS: {precipitationCoverageStations.crs}")
    print(f"Sediment coverage CRS: {sedimentCoverage.crs}")
    
    # Check if 'stationId' or 'station_id' column exists
    id_column = 'station_id'
    if 'stationId' in precipitationCoverageStations.columns:
        id_column = 'stationId'
    
    print(f"Using {id_column} to identify stations.")
    
    # Get the stations matching the stationId
    matching_stations = precipitationCoverageStations[precipitationCoverageStations[id_column] == stationId]
    
    # Check if we found any matching stations
    if matching_stations.empty:
        print(f"Warning: No station found with ID {stationId}")
        return []
    
    # Get the geometry of the station
    station_geometry = matching_stations.geometry.iloc[0]
    
    # Print geometry information for debugging
    print(f"Station geometry type: {station_geometry.geom_type}")
    print(f"Station geometry bounds: {station_geometry.bounds}")
    
    # Ensure both datasets use the same CRS
    if precipitationCoverageStations.crs != sedimentCoverage.crs:
        #print(f"CRS mismatch! Reprojecting sediment coverage to {precipitationCoverageStations.crs}")
        sedimentCoverage = sedimentCoverage.to_crs(precipitationCoverageStations.crs)
    
    # Check if the geometries are valid
    if not station_geometry.is_valid:
        print("Station geometry is invalid! Attempting to fix...")
        station_geometry = station_geometry.buffer(0)
    
    # Use buffer to account for possible precision issues
    # This creates a small buffer around the station geometry to increase chances of intersection
    buffered_geometry = station_geometry.buffer(1)  # 1 meter buffer

    # Try with the buffered geometry first
    sediment_types_buffered = sedimentCoverage[sedimentCoverage.intersects(buffered_geometry)]
    
    if not sediment_types_buffered.empty:
        print(f"Found {len(sediment_types_buffered)} sediment features using buffered geometry")
        sediment_types = sediment_types_buffered
    else:
        # If buffered approach fails, try with original geometry
        sediment_types = sedimentCoverage[sedimentCoverage.intersects(station_geometry)]
        if sediment_types.empty:
            # Check if any sediment polygons are nearby
            # This helps diagnose if the issue is with projection or data
            buffer_distance = 1000  # 1 km
            large_buffer = station_geometry.buffer(buffer_distance)
            nearby_sediments = sedimentCoverage[sedimentCoverage.intersects(large_buffer)]
            
            if not nearby_sediments.empty:
                print(f"Found {len(nearby_sediments)} sediment features within {buffer_distance}m")
                print("The issue might be with projection or precision")
            else:
                print(f"No sediment features found even within {buffer_distance}m")
                print("The station might be outside the sediment coverage area")
            
            # Print a sample of sediment geometries to compare
            if not sedimentCoverage.empty:
                sample_sediment = sedimentCoverage.iloc[0]
                print(f"Sample sediment bounds: {sample_sediment.geometry.bounds}")
            
            return []
    
    # Check if 'tsym' column exists
    if 'tsym' not in sediment_types.columns:
        # Try to find a suitable column for soil types
        print(f"Available columns in sediment data: {sediment_types.columns.tolist()}")
        soil_type_columns = [col for col in sediment_types.columns if 'type' in col.lower() or 'sym' in col.lower() or 'soil' in col.lower()]
        
        if soil_type_columns:
            soil_type_column = soil_type_columns[0]
            print(f"Using '{soil_type_column}' instead of 'tsym' for soil types")
            soil_types = sediment_types[soil_type_column].unique().tolist()
        else:
            print(f"No suitable soil type column found")
            return []
    else:
        # Extract soil types from the filtered data
        soil_types = sediment_types['tsym'].unique().tolist()
    
    print(f"Found soil types: {soil_types}")
    return soil_types

def gather_soil_types(purculation_mapping):
    """
    Create a dictionary of soil types with their average percolation rates.
    
    Parameters:
    -----------    
    purculation_mapping : dict
        Dictionary with soil types as keys and min/max percolation rates
        
    Returns:
    --------
    dict
        Dictionary with soil types as keys and average percolation rates as values
    
    Called by: load_process_data()
    """
    # Take perculation Keys and the min and max / 2 and add to a dict
    soil_types = {}
    for key, value in purculation_mapping.items():
        min = 0.0001 if value['min'] == 0 else value['min']
        max = 0.9999 if value['max'] == 1 else value['max']
            
        soil_types[key] = (min + max) / 2
    return soil_types

###########################################
# SECTION 3: WATER CALCULATIONS           #
###########################################

def calculate_water_on_ground(df, soil_types, absorbtions, station):
    """
    Calculate water on ground for specific soil types and station.

    Parameters:
    -----------    
    df : pandas.DataFrame
        Dataframe containing precipitation data with a 'Nedbor' column
    soil_types : list
        List of soil types to calculate water on ground for
    absorbtions : dict
        Dictionary with soil types and their absorption rates
    station : str
        Station ID to use in column naming
    
    Returns:
    --------
    pandas.DataFrame: Dataframe with water on ground values for the specified soil types
    
    Called by: load_process_data()
    """
    # Get precipitation values as numpy array for faster calculations
    precip_array = df['Nedbor'].values
    n = len(precip_array)
    
    # Process all soil types at once using numpy operations
    soil_type_data = {}
    new_columns = {}  # Dict to collect all columns before creating DataFrame
    valid_soil_types = [st for st in soil_types if st in absorbtions]
    
    if not valid_soil_types:
        print(f"No valid soil types with known absorption rates for station {station}")
        return df.copy()
        
    # Pre-allocate numpy arrays for all calculations to avoid memory allocations in loops
    for soil_type in valid_soil_types:
        rate = absorbtions[soil_type] 
        soil_type_data[soil_type] = {
            'rate': rate,
            'wog_array': np.zeros(n),
            'observed': np.zeros(n, dtype=int),
            'tte': np.full(n, n),  # Fill with max value initially
            'duration': np.zeros(n, dtype=int)
        }
    
    # Parallel WOG calculation for each soil type using vectorized operations where possible
    for soil_type, data in soil_type_data.items():
        rate = data['rate']
        wog = data['wog_array']
        
        # First time step
        wog[0] = max(0, precip_array[0])
        
        # Vectorized recurrence relation using a cumulative approach
        for i in range(1, n):
            wog[i] = max(0, wog[i-1] * (1 - rate) + precip_array[i])
        
        # Calculate observed state (> threshold)
        # wog_window = np.convolve(wog, np.ones(3)/3, mode='same')  # 3-hour window
        # data['observed'] = (wog_window > 5).astype(int)

        data['observed'] = (wog > 5).astype(int) # CHANGE HERE!
        
        # Find event indices
        event_indices = np.where(data['observed'] == 1)[0]
        # First pass: Calculate time until next event (survival analysis approach)
        tte = np.full(n, n)  # Default to maximum for censored observations
        durations = np.full(n, n)  # Default to maximum

        # Mark events with time-to-event = 0
        tte[event_indices] = 0

        # For each pair of events, calculate time between them
        for i in range(len(event_indices)-1):
            start_idx = event_indices[i]
            end_idx = event_indices[i+1]
            time_between = end_idx - start_idx
            
            # Fill in counting up from 1 at non-event to event time at event
            for j in range(start_idx+1, end_idx):
                tte[j] = end_idx - j
            
            # Store the duration (time until next event)
            durations[start_idx:end_idx] = np.arange(1, time_between+1)

        # For observations after the last event, they're all censored
        if len(event_indices) > 0:
            last_event = event_indices[-1]
            durations[last_event+1:] = np.arange(1, n-last_event)

        # Store calculated values
        data['tte'] = tte
        data['duration'] = durations
            
        # Add columns to the dictionary
        new_columns[f'{station}_WOG_{soil_type}'] = data['wog_array']
        new_columns[f"{station}_{soil_type}_observed"] = data['observed']
        new_columns[f'{station}_{soil_type}_TTE'] = data['tte']
        new_columns[f'{station}_{soil_type}_duration'] = data['duration']
    
    # Create new DataFrame with all columns at once
    new_df = pd.DataFrame(new_columns, index=df.index)
    
    # Combine with original data
    result_df = pd.concat([df, new_df], axis=1)
    
    return result_df

###########################################
# SECTION 4: DATA LOADING AND SAVING      #
###########################################

def load_process_data(coverage_data=None, sediment_data=None):
    """
    Load precipitation data and calculate water-on-ground values for different soil types.
    
    Parameters:
    -----------
    coverage_data : GeoDataFrame, optional
        Pre-loaded precipitation coverage data
    sediment_data : GeoDataFrame, optional
        Pre-loaded sediment coverage data
    
    Returns:
    --------
    pandas.DataFrame: Processed data with soil type observations and durations
    
    Called by: main
    Calls: gather_soil_types(), sediment_types_for_station(), calculate_water_on_ground(), save_preprocessed_data()
    """
    try:
        # Load the data
        print("\nLoading precipitation imputed data...")
        df = pd.read_parquet("data/raw/precipitation_imputed_data.parquet")
        # print(f"Loaded precipitation data with columns: {df.columns.tolist()[:]}...")
        print(f"This is a total number of columns: {len(df.columns)}, which is the number of stations")
        # each row is an hour of precipitation data for each station
        # save as csv for easier debugging
        df.to_csv("data/raw/precipitation_imputed_data.csv", index=False)
        print(f"Precipitation data shape: {df.shape}") #(262783, 86)

        # total number of nans across all columns (all stations)
        print(f"Total number of NaNs in the data: {df.isna().sum().sum()}")

        # look column-wise for the number of nans
        stations_with_most_nans = df.isna().sum().sort_values(ascending=False)
        print(f"Top 5 Stations with the most NaNs in procent of that station: {(stations_with_most_nans / len(df) * 100).head(5)}")
        # here we can divide by len(df) because all columns have the same length

        # before clipping to remove extreme values, we need to check the data
        # Check for extreme values in the data
        # print(f"Precipitation data summary:\n{df.describe()}")

        # def length before clipping - which just replaces values lower than 0 with 0 and values higher than 60 with 60
        # clip does not replace Nans, only limits existing values 
        df = df.clip(lower=0, upper=100) 
        # check min and max values


        #https://international.kk.dk/sites/default/files/2021-09/Cloudburst%20Management%20plan%202010.pdf?utm_source=chatgpt.com
        # precipitation measured close to 100 mm in one hour.

        #https://web.archive.org/web/20140913151609/http://vejret.tv2.dk/artikel/id-32909558:et-af-de-kraftigste-regnvejr-nogensinde.html
        # over 100mm in 24 hours and private measurements for 160mm in 124 hours

        #https://ui.adsabs.harvard.edu/abs/2021AGUFMGC45G0892C/abstract
        #Between 90 and 135 mm of precipitation in less than 2 hours was recorded

        #https://vejr.tv2.dk/2019-12-28-her-er-de-danske-vejrrekorder-fra-de-seneste-10-aar
        # Here the record is 63mm in 30mins

        #https://vejr.tv2.dk/2016-07-02-husker-du-vejret-den-2-juli-2011-historisk-skybrud-ramte-koebenhavn
        # Kraftig regn er, når der falder mere end 24 millimeter regn over en periode på maksimalt seks timer.
        # Skybrud er, når der falder mere end 15 millimeter regn over en periode på maksimalt 30 minutter.
        # def length after clipping

        # Use provided coverage data or load from file
        if coverage_data is not None:
            precipitationCoverageStations = coverage_data
            print(f"Using provided precipitation coverage with {len(precipitationCoverageStations)} stations")
        else:
            print("Loading precipitation coverage stations...")
            precipitationCoverageStations = gu.load_geojson("precipitation_coverage.geojson")
            print(f"Loaded precipitation coverage with {len(precipitationCoverageStations)} stations")
        
        # Use provided sediment data or load from file
        if sediment_data is not None:
            sedimentCoverage = sediment_data
            print(f"Using provided sediment coverage with {len(sedimentCoverage)} features")
        else:
            print("\nLoading sediment coverage...")
            sedimentCoverage = gu.load_geojson("Sediment_wgs84.geojson")
            print(f"Loaded sediment coverage with {len(sedimentCoverage)} features")

        # Get absorption rates for each soil type
        absorbtions = gather_soil_types(pm.percolation_rates_updated)
        print(f"Gathered absorption rates for {len(absorbtions)} soil types")
        
        print(f"\nPrecipitaion dataset columns: {df.columns.tolist()}")
        stations_to_process = df.columns

        # For each station in the data, calculate the water on ground for each soil type
        for station in stations_to_process:
            print(f"Processing station {station}...")
            df_station = df[[station]].copy() # this is a single column dataframe with the precipitation data for this station
            
            # Rename the station name column to 'Nedbor' (precipitation) for consistency
            df_station.rename(columns={station: 'Nedbor'}, inplace=True)

            # length before dropping NaN values
            pre_drop_nans = len(df_station)
            print(f"  • Precipitation data {station} length before dropping NaN: {len(df_station)}")
            df_station.dropna(inplace=True) 
            # we remove the rows with NaN values, because they are not useful for our calculations 
            # as we want to calculate the water on ground only for the rows with precipitation data
            print(f"  • Removed {pre_drop_nans - len(df_station)} NaN values (procent {(pre_drop_nans - len(df_station)) / pre_drop_nans * 100:.2f}%)")
            
            if df_station.empty:
                print(f"No data for station {station}, skipping...")
                continue
            
            # Get soil types for this station
            sediment_types = sediment_types_for_station(station, precipitationCoverageStations, sedimentCoverage)
            
            if not sediment_types:
                print(f"No sediment types found for station {station}, skipping...")
                continue
                
            print(f"Found {len(sediment_types)} sediment types for station {station}")
            
            # Calculate water on ground for each soil type
            try:
                df_processed = calculate_water_on_ground(df_station, sediment_types, absorbtions, station)
                # Add processed columns to results (excluding 'Nedbor')
                result_columns = df_processed.drop(columns=['Nedbor'], errors='ignore')
                if not result_columns.empty:
                    save_preprocessed_data(result_columns, f"data/processed/survival_data_{station}.csv")
                    print(f"Saved processed data for station {station}")
            except Exception as e:
                print(f"ERROR processing station {station}: {e}")
                continue
        
        # Combine all result DataFrames at once
        return None  # Return None to indicate completion
        
    except Exception as e:
        print(f"ERROR in load_process_data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

def save_preprocessed_data(survival_df, output_path="data/processed/survival_data.csv"):
    """
    Save the processed survival data to a CSV file.
    
    Parameters:
    -----------    
    survival_df : DataFrame
        DataFrame containing survival data for different soil types and stations
    output_path : str
        Path to save the combined CSV file
    
    Called by: load_process_data()
    """
    import os
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save to Parquet for better performance
    try:
        survival_df.to_parquet(output_path.replace('.csv', '.parquet'), index=False)
        print(f"Saved preprocessed survival data to {output_path.replace('.csv', '.parquet')}")
    except Exception as e:
        print(f"ERROR saving to Parquet: {e}")
        print("Falling back to CSV format")
        try:
            survival_df.to_csv(output_path, index=False)
            print(f"Saved preprocessed survival data to {output_path}")
        except Exception as e:
            print(f"ERROR saving to CSV: {e}")
            print("Failed to save preprocessed data")
            return None
    
    return survival_df

def load_saved_data(file_path="data/processed/survival_data.csv"):
    """
    Load previously saved preprocessed data.
    
    Parameters:
    -----------    
    file_path : str
        Path to the saved data file
    
    Returns:
    --------
    dict
        Dictionary with soil types as keys and survival dataframes as values
    
    Called by: main
    """

    import os

    # load parquet file if it exists
    try:
        if os.path.exists(file_path.replace('.csv', '.parquet')):
            survival_df = pd.read_parquet(file_path.replace('.csv', '.parquet'))
            print(f"Loaded preprocessed data from {file_path.replace('.csv', '.parquet')}")
            return survival_df
    except Exception as e:
        print(f"ERROR loading Parquet data: {e}")

    try:
        # Load the combined dataframe
        survival_df = pd.read_csv(file_path)
        
        return survival_df
        
    except Exception as e:
        print(f"ERROR loading data from {file_path}: {e}")
        return None

###########################################
# MAIN EXECUTION                          #
###########################################

if __name__ == "__main__":

    # TODO uncomment this
    # Step 1: Create coverage areas for precipitation stations
    coverage_geojson_gdf, stations_gdf = create_full_coverage()
    # this functions saves the coverage_geojson_gdf to a file called precipitation_coverage.geojson
    if coverage_geojson_gdf is None:
        print("ERROR No valid coverage data created. Attemption to load from file.")
        try:
            coverage_geojson_gdf = gu.load_geojson("precipitation_coverage.geojson")
            print("Loaded coverage data from file.")
        except Exception as e:
            print(f"ERROR loading coverage data from file: {e}")
            print("Exiting.")
            exit(1)
    
    # Step 2: Load sediment data - this was the layer that was exported from the QGIS project
    print("Loading Sediment_wgs84.geojson...")
    sedimentCoverage = gu.load_geojson("Sediment_wgs84.geojson")
    if sedimentCoverage is None:
        print("No valid sediment data loaded. Exiting.")
        exit(1)
    
    # Step 3: Process precipitation and soil data
    # - this gathers the soil types
    # For each station, it calculates
    # * sediment types for the given station
    # * water on ground for each soil type
    # * saves the processed data to a CSV file
    load_process_data(coverage_data=coverage_geojson_gdf, sediment_data=sedimentCoverage)

    # Step 4: TESTING - Load and display sample processed data for a specific station
    station_id = '06058'  # Example station ID
    df = load_saved_data(f'data/processed/survival_data_{station_id}.csv')
    if df is not None:
        print(f"\nSample data for station {station_id}:")
        print(df.head())
    else:
        print(f"No data found for station {station_id}")