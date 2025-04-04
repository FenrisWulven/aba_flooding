import json
import geopandas as gpd
import pandas as pd
import numpy as np
from bokeh.models import GeoJSONDataSource, LinearColorMapper
from bokeh.palettes import Viridis256
import geopandas as gpd
import os
from pathlib import Path
from pyproj import Transformer
from shapely.ops import transform

# WIll take the a model and geodata and apply the survival function on the data so it is in geojson format for the map

def get_data_dir():
    """Return path to the raw data directory."""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    return data_dir

def load_geojson(file_name):
    """Load a GeoJSON file from the data directory."""
    data_dir = get_data_dir()
    file_path = os.path.join(data_dir, file_name)
    return gpd.read_file(file_path)

def load_gpkg(file_name, layer=None):
    """Load a GeoPackage file from the data directory, with optional layer name."""
    data_dir = get_data_dir()
    file_path = os.path.join(data_dir, file_name)
    if layer:
        return gpd.read_file(file_path, layer=layer)
    return gpd.read_file(file_path)

def load_terrain_data(file_name):
    """Load terrain data from GeoJSON or GPKG file."""
    if file_name.endswith('.geojson'):
        return load_geojson(file_name)
    elif file_name.endswith('.gpkg'):
        return load_gpkg(file_name)
    else:
        raise ValueError(f"Unsupported file format for {file_name}")

def wgs84_to_web_mercator(df):
    """Convert GeoDataFrame from WGS84 to Web Mercator projection."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    # Create new geometry column with transformed coordinates
    df = df.copy()
    df['geometry'] = df['geometry'].apply(
        lambda geom: transform(lambda x, y: transformer.transform(x, y), geom)
    )
    return df

def gdf_to_geojson(gdf):
    """Convert GeoDataFrame to GeoJSON format for Bokeh."""
    # Make a copy to avoid modifying the original
    gdf = gdf.copy()
    
    # Convert any datetime columns to strings
    for col in gdf.select_dtypes(include=['datetime64[ns]']).columns:
        gdf[col] = gdf[col].astype(str)
    
    # Also check for individual Timestamp objects in the DataFrame
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
            gdf[col] = gdf[col].apply(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
    
    # Convert to GeoJSON
    return json.dumps(json.loads(gdf.to_json()))



def survival_layer(terrain_geojson, sediment_geojson, year, model=None, terrain_df=None, sediment_df=None, preserve_topology=False):
    """Creates a flood risk visualization layer using sediment data as primary and terrain as secondary feature."""
    print("Creating flood visualization layer...")
    
    if sediment_df is None or len(sediment_df) == 0:
        print("No valid sediment data provided")
        return None
    
    try:
        # Make a clean copy of the sediment data
        flood_data = sediment_df.copy()
        
        # Sample if the dataset is too large (for performance)
        sample_size = len(flood_data)
        if len(flood_data) > sample_size:
            # Use spatial sampling to maintain geographic distribution
            flood_data = flood_data.sample(sample_size, random_state=42)
        
        print(f"Processing {len(flood_data)} sediment features")
        
        # Add synthetic flood probability data
        # This will be replaced with a real model in the future
        if model is not None:
            try:
                flood_data["flood_probability"] = model.predict_flood_risk(flood_data, years=year)
            except Exception as e:
                print(f"Error using prediction model: {str(e)}")
                # Fallback to random values
                flood_data["flood_probability"] = np.random.beta(2, 5, size=len(flood_data))
        else:
            # Use random values if no model is provided
            flood_data["flood_probability"] = np.random.beta(2, 5, size=len(flood_data))
                
        # Incorporate terrain data if available
        if terrain_df is not None and len(terrain_df) > 0:
            print("Incorporating terrain data as secondary feature...")
            # Here you might join or merge with terrain data
            # This is a placeholder - implement specific join logic based on your needs
            
            # Add elevation from terrain if missing in sediment data
            if 'elevation' not in flood_data.columns and 'elevation' in terrain_df.columns:
                # This is simplified - in practice you would need a spatial join
                flood_data['elevation'] = terrain_df['elevation'].mean()
        
        # Add default elevation if missing
        if 'elevation' not in flood_data.columns:
            if 'z' in flood_data.columns:
                flood_data['elevation'] = flood_data['z']
            else:
                flood_data['elevation'] = 2.0  # Default value
        
        # Ensure all required columns exist
        required_columns = ['geometry', 'flood_probability']
        for col in required_columns:
            if col not in flood_data.columns:
                print(f"Missing required column: {col}")
                return None
        
        # Convert any datetime columns to strings to avoid JSON serialization errors
        for col in flood_data.select_dtypes(include=['datetime64[ns]']).columns:
            flood_data[col] = flood_data[col].astype(str)
        
        # Also check for individual Timestamp objects in the DataFrame
        for col in flood_data.columns:
            if pd.api.types.is_object_dtype(flood_data[col]):
                # Check if column contains any Timestamp objects
                if flood_data[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    flood_data[col] = flood_data[col].apply(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
        
        # Convert to GeoJSON for Bokeh
        print("Converting to GeoJSON source...")
        flood_json = flood_data.to_json()
        
        # Validate the GeoJSON
        json_data = json.loads(flood_json)
        feature_count = len(json_data.get('features', []))
        print(f"GeoJSON contains {feature_count} features")
        
        if feature_count == 0:
            print("Warning: GeoJSON has no features!")
            return None
            
        # Create the final data source
        source = GeoJSONDataSource(geojson=flood_json)
        return source
        
    except Exception as e:
        print(f"Error creating flood layer: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def simple_flood_model(data, year):
    return np.random.rand(len(data))