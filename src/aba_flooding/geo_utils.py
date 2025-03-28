import json
import geopandas as gpd
import pandas as pd
import numpy as np
from bokeh.models import GeoJSONDataSource, LinearColorMapper
from bokeh.palettes import Viridis256

class SurvivalModel:
    def __init__(self, model):
        self.model = model
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def survival_layer(terrain_geojson, sediment_geojson, year, model=None, terrain_df=None, sediment_df=None, preserve_topology=False):
    """Ultra-simplified function that just returns a tiny subset of terrain data with random flood values"""
    print("Creating minimal flood visualization...")
    
    if terrain_df is None:
        print("No terrain data provided")
        return GeoJSONDataSource(geojson='{"type":"FeatureCollection","features":[]}')
    
    # Get a small sample with minimal processing
    sample_size = 500  # Even smaller sample
    print(f"Taking sample of {sample_size} features")
    
    try:
        # Make a copy to avoid modifying the original
        terrain_sample = terrain_df.copy()
        
        # Take a small random sample
        if len(terrain_sample) > sample_size:
            terrain_sample = terrain_sample.sample(sample_size, random_state=42)
        
        # Verify we have data
        print(f"Sample contains {len(terrain_sample)} features")
        
        # Ensure geometry is valid
        terrain_sample = terrain_sample[~terrain_sample.geometry.is_empty]
        print(f"After removing empty geometries: {len(terrain_sample)} features")
        
        # Add random flood probabilities
        terrain_sample["flood_probability"] = np.random.rand(len(terrain_sample))
        
        # Add a default elevation if it doesn't exist
        if 'elevation' not in terrain_sample.columns:
            print("Adding default elevation column")
            terrain_sample['elevation'] = np.random.uniform(0, 10, len(terrain_sample))
        
        # Check if we have any data left
        if len(terrain_sample) == 0:
            print("No valid features remain!")
            return GeoJSONDataSource(geojson='{"type":"FeatureCollection","features":[]}')
            
        # Convert directly to GeoJSON
        print("Converting to GeoJSON...")
        result_geojson = terrain_sample.to_json()
        
        # Verify GeoJSON is not empty
        json_obj = json.loads(result_geojson)
        feature_count = len(json_obj.get('features', []))
        print(f"GeoJSON contains {feature_count} features")
        
        if feature_count == 0:
            print("Warning: GeoJSON has no features!")
            
        # Create the GeoJSONDataSource
        source = GeoJSONDataSource(geojson=result_geojson)
        
        # Debug using the geojson property instead of data
        print(f"Source has geojson: {source.geojson is not None and len(source.geojson) > 0}")
        
        return source
        
    except Exception as e:
        print(f"Error in creating flood layer: {e}")
        import traceback
        traceback.print_exc()
        return GeoJSONDataSource(geojson='{"type":"FeatureCollection","features":[]}')

def simple_flood_model(data, year):
    return np.random.rand(len(data))