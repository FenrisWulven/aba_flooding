import geopandas as gpd
import pydeck as pdk
from shapely.geometry import box
import os
from pathlib import Path
import pandas as pd

def get_data_dir():
    """Return path to the raw data directory."""
    try: 
        current_file = Path(__file__)
    except:
        current_file = Path(os.getcwd())
    project_root = current_file.parent.parent.parent
    data_dir = project_root / "aba_flooding" / "data" / "raw"
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
        # geopandas.read_file(filename, bbox=None, mask=None, columns=None, rows=None, engine=None, **kwargs)
        return gpd.read_file(file_path, mask=layer)
    else:
        return gpd.read_file(file_path)

def load_terrain_data(file_name):
    """Load terrain data from GeoJSON or GPKG file."""
    if file_name.endswith('.geojson'):
        return load_geojson(file_name)
    elif file_name.endswith('.gpkg'):
        return load_gpkg(file_name)
    else:
        raise ValueError(f"Unsupported file format for {file_name}")



# Run visualization
if __name__ == "__main__":
    terrain_file = "Sediment_wgs84.geojson"
    # Load terrain data
    
    # Show the first few rows of the loaded data
    terrain_data = load_terrain_data(terrain_file)
    
    # Plot with bokeh
