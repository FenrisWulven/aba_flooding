import json
import geopandas as gpd
import pandas as pd
import numpy as np
from bokeh.models import GeoJSONDataSource, LinearColorMapper
from bokeh.palettes import Viridis256
import os
from pathlib import Path
from pyproj import Transformer
from shapely.ops import transform
import fiona

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
    try:
        # First try standard geopandas approach
        return gpd.read_file(file_path)
    except Exception as e:
        print(f"Standard GeoPandas read failed: {e}")
        try:
            # Try alternate method with fiona
            import fiona
            with fiona.open(file_path, 'r') as src:
                crs = src.crs
                features = list(src)
            
            # Convert to GeoDataFrame
            import shapely.geometry
            geoms = [shapely.geometry.shape(feature['geometry']) for feature in features]
            properties = [feature['properties'] for feature in features]
            
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(properties, geometry=geoms, crs=crs)
            return gdf
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            raise e

def load_gpkg(file_name, layer=None):
    """Load a GeoPackage file from the data directory, with optional layer name."""
    data_dir = get_data_dir()
    file_path = os.path.join(data_dir, file_name)
    
    if layer:
        return gpd.read_file(file_path, layer=layer)
    else:
        # Try to get available layers first
        try:
            layers = fiona.listlayers(file_path)
            if len(layers) > 0:
                print(f"Available layers in {file_name}: {layers}")
                return gpd.read_file(file_path, layer=layers[0])
            else:
                return gpd.read_file(file_path)
        except Exception as e:
            print(f"Error getting layers: {e}")
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

def convert_to_serializable(obj):
    """Convert non-serializable objects to JSON serializable types."""
    import numpy as np
    import pandas as pd
    from datetime import datetime, date
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return str(obj)

def gdf_to_geojson(gdf):
    """
    Convert a GeoDataFrame to GeoJSON format.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to convert
    
    Returns:
    --------
    str
        GeoJSON string
    """
    import json
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Preprocess the dataframe to handle problematic columns
    df_copy = gdf.copy()
    
    # Convert all timestamp columns to strings
    for col in df_copy.columns:
        if col != 'geometry':
            # Check if column has timestamp data
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)
            # Convert any numpy data type columns to native Python types
            elif pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].apply(
                    lambda x: float(x) if pd.api.types.is_float_dtype(type(x)) else 
                    int(x) if pd.api.types.is_integer_dtype(type(x)) else x
                )
    
    # Use a custom serialization approach
    try:
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                return json.JSONEncoder.default(self, obj)
        
        # First convert to GeoJSON dict
        geo_dict = json.loads(df_copy.to_json())
        
        # Then serialize with custom encoder
        return json.dumps(geo_dict, cls=CustomEncoder)
        
    except Exception as e:
        print(f"Error in GeoJSON serialization: {e}")
        
        # Fallback approach - manually build GeoJSON
        features = []
        for idx, row in df_copy.iterrows():
            try:
                properties = {}
                for col in df_copy.columns:
                    if col != 'geometry':
                        val = row[col]
                        properties[col] = convert_to_serializable(val)
                
                geometry = row['geometry'].__geo_interface__
                features.append({
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometry
                })
            except Exception as feat_e:
                print(f"Error processing feature {idx}: {feat_e}")
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return json.dumps(geojson)
