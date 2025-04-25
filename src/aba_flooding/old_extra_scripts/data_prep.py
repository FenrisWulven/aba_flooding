import os
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pydeck as pdk
import json
from shapely.geometry import mapping
from IPython.display import display

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

def load_sediment_data(file_name):
    """Load sediment data from GeoJSON or GPKG file."""
    if file_name.endswith('.geojson'):
        return load_geojson(file_name)
    elif file_name.endswith('.gpkg'):
        return load_gpkg(file_name)
    else:
        raise ValueError(f"Unsupported file format for {file_name}")

def load_terrain_data(file_name):
    """Load terrain data from GeoJSON or GPKG file."""
    if file_name.endswith('.geojson'):
        return load_geojson(file_name)
    elif file_name.endswith('.gpkg'):
        return load_gpkg(file_name)
    else:
        raise ValueError(f"Unsupported file format for {file_name}")

def prepare_for_ml(gdf):
    """
    Prepare a GeoDataFrame for machine learning by:
    1. Converting geometry to features (centroids, area, etc.)
    2. Handling categorical variables
    3. Scaling numerical features
    """
    # Create a copy to avoid modifying the original
    df = gdf.copy()
    
    # Extract features from geometry
    if 'geometry' in df.columns:
        # Add centroid coordinates
        df['centroid_x'] = df.geometry.centroid.x
        df['centroid_y'] = df.geometry.centroid.y
        
        # Add area if polygons
        if any(df.geometry.type.isin(['Polygon', 'MultiPolygon'])):
            df['area'] = df.geometry.area
        
        # Add length if lines
        if any(df.geometry.type.isin(['LineString', 'MultiLineString'])):
            df['length'] = df.geometry.length
    
    # Handle categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'geometry':  # Skip geometry column
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # Drop geometry column for ML
    if 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
    
    # Remove any remaining non-numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Scale numerical features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    return df_scaled

def split_features_target(df, target_column):
    """Split the dataframe into features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def load_and_prepare_data(sediment_file, terrain_file=None, target_column=None):
    """
    Load and prepare geospatial data for machine learning.
    
    Parameters:
    sediment_file (str): Filename of sediment data
    terrain_file (str, optional): Filename of terrain data
    target_column (str, optional): Column to use as target variable
    
    Returns:
    tuple: Features and target (if target_column is provided)
    """
    # Load sediment data
    sediment_data = load_sediment_data(sediment_file)
    
    # Load terrain data if provided
    if terrain_file:
        terrain_data = load_terrain_data(terrain_file)
        # Spatial join of sediment and terrain data
        # Adjust the how parameter based on your needs
        combined_data = gpd.sjoin(sediment_data, terrain_data, how='left', predicate='intersects')
    else:
        combined_data = sediment_data
    
    # Prepare for machine learning
    ml_ready_data = prepare_for_ml(combined_data)
    
    # Split into features and target if target column is provided
    if target_column and target_column in ml_ready_data.columns:
        return split_features_target(ml_ready_data, target_column)
    else:
        return ml_ready_data

def gdf_to_pydeck(gdf, color_by=None, color_scale=None):
    """
    Convert a GeoDataFrame to a format suitable for pydeck visualization.
    
    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to convert
    color_by (str, optional): Column name to use for color scaling
    color_scale (list, optional): List of colors to use for the scale
    
    Returns:
    tuple: (data for pydeck, column for coloring if provided)
    """
    # Create a copy to avoid modifying the original
    gdf = gdf.copy()
    
    # Handle different geometry types
    if len(gdf) == 0:
        return pd.DataFrame(), None
    
    # For polygons, we need to convert to GeoJSON format
    if any(gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])):
        # Create a GeoJSON feature for each row
        features = []
        for _, row in gdf.iterrows():
            geom = mapping(row.geometry)
            properties = {col: row[col] for col in gdf.columns if col != 'geometry'}
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": properties
            })
        
        # Return a dictionary that can be used directly with pydeck's GeoJsonLayer
        return {"type": "FeatureCollection", "features": features}, color_by
    
    # For points, just extract coordinates
    elif any(gdf.geometry.type.isin(['Point', 'MultiPoint'])):
        df = gdf.copy()
        df['lon'] = df.geometry.x
        df['lat'] = df.geometry.y
        if 'geometry' in df.columns:
            df = df.drop(columns=['geometry'])
        return df, color_by
    
    # For lines, convert to GeoJSON as well
    elif any(gdf.geometry.type.isin(['LineString', 'MultiLineString'])):
        features = []
        for _, row in gdf.iterrows():
            geom = mapping(row.geometry)
            properties = {col: row[col] for col in gdf.columns if col != 'geometry'}
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": properties
            })
        return {"type": "FeatureCollection", "features": features}, color_by
    
    return pd.DataFrame(), None

def visualize_terrain_data(terrain_data, elevation_scale=20):
    """
    Create a pydeck visualization for terrain data.
    
    Parameters:
    terrain_data (GeoDataFrame): The terrain data to visualize
    elevation_scale (int): Scale factor for elevation
    
    Returns:
    pydeck.Deck: A deck object for rendering
    """
    # Calculate centroids correctly in the original projected CRS
    original_crs = terrain_data.crs
    centroids = terrain_data.geometry.centroid
    center_x = centroids.x.mean()
    center_y = centroids.y.mean()
    
    # Create a single point for the center and convert to WGS84
    center_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([center_x], [center_y]), crs=original_crs)
    center_point_wgs84 = center_point.to_crs(epsg=4326)
    center_lon = center_point_wgs84.geometry.x[0]
    center_lat = center_point_wgs84.geometry.y[0]
    
    # Convert to pydeck format
    pydeck_data, color_column = gdf_to_pydeck(terrain_data, color_by='elevation')
    
    # Create a polygon layer for terrain
    if isinstance(pydeck_data, dict) and pydeck_data.get("type") == "FeatureCollection":
        layer = pdk.Layer(
            'GeoJsonLayer',
            data=pydeck_data,
            opacity=0.8,
            stroked=False,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation='properties.elevation',
            get_fill_color='[0, 100, 200, 160]',  # Blue color
            get_line_color='[0, 0, 0, 50]',
            elevation_scale=elevation_scale,
            pickable=True,
            auto_highlight=True
        )
    else:
        # Fallback to ScatterplotLayer if not polygons
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=pydeck_data,
            get_position=['lon', 'lat'],
            get_color='[0, 100, 200, 160]',
            get_radius=100,
            pickable=True
        )
    
    # Set up the deck with adjusted zoom level
    view_state = pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=6,  # Zoomed out to show all of Denmark
        pitch=30,  # Less steep angle for better overview
        bearing=0
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Elevation: {properties.elevation}m"},
        map_style='mapbox://styles/mapbox/light-v9'
    )

def visualize_combined_data(terrain_data, sediment_data=None):
    """
    Create a pydeck visualization for terrain and sediment data.
    
    Parameters:
    terrain_data (GeoDataFrame): The terrain data to visualize
    sediment_data (GeoDataFrame, optional): The sediment data to visualize
    
    Returns:
    pydeck.Deck: A deck object for rendering
    """
    # Calculate centroids correctly in the original projected CRS
    original_crs = terrain_data.crs
    centroids = terrain_data.geometry.centroid
    center_x = centroids.x.mean()
    center_y = centroids.y.mean()
    
    # Create a single point for the center and convert to WGS84
    center_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([center_x], [center_y]), crs=original_crs)
    center_point_wgs84 = center_point.to_crs(epsg=4326)
    center_lon = center_point_wgs84.geometry.x[0]
    center_lat = center_point_wgs84.geometry.y[0]
    
    layers = []
    
    # Add terrain layer
    terrain_pydeck_data, _ = gdf_to_pydeck(terrain_data, color_by='elevation')
    if isinstance(terrain_pydeck_data, dict) and terrain_pydeck_data.get("type") == "FeatureCollection":
        terrain_layer = pdk.Layer(
            'GeoJsonLayer',
            data=terrain_pydeck_data,
            opacity=0.8,
            stroked=False,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation='properties.elevation',
            get_fill_color='[0, 100, 200, 160]',
            get_line_color='[0, 0, 0, 50]',
            elevation_scale=50,
            pickable=True,
            auto_highlight=True
        )
        layers.append(terrain_layer)
    
    # Add sediment layer if provided
    if sediment_data is not None:
        sediment_pydeck_data, _ = gdf_to_pydeck(sediment_data)
        if isinstance(sediment_pydeck_data, dict) and sediment_pydeck_data.get("type") == "FeatureCollection":
            sediment_layer = pdk.Layer(
                'GeoJsonLayer',
                data=sediment_pydeck_data,
                opacity=0.8,
                stroked=True,
                filled=True,
                extruded=False,
                get_fill_color='[200, 30, 0, 160]',  # Red color
                get_line_color='[200, 30, 0, 200]',
                get_line_width=2,
                pickable=True,
                auto_highlight=True
            )
            layers.append(sediment_layer)
    
    # Set up the deck with adjusted zoom level
    view_state = pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=6,  # Zoomed out to show all of Denmark
        pitch=30,  # Less steep angle for better overview
        bearing=0
    )
    
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Data properties: {properties}"},
        map_style='mapbox://styles/mapbox/light-v9'
    )

def main():
    """
    Demonstrate the data preparation functionality.
    This function loads sample data, prepares it for ML, and displays information.
    """
    print("Data Preparation Module Demo")
    print("-" * 50)
    
    # Updated to use actual files in the data directory
    terrain_file = "Terrain.geojson"
    sediment_file = "Sediment.geojson"  
    
    print(f"Data directory path: {get_data_dir()}")
    
    # Try to work with the terrain data that we know exists
    try:
        print(f"\nLoading terrain data from: {terrain_file}")
        terrain_data = load_terrain_data(terrain_file)
        print(f"Loaded terrain data shape: {terrain_data.shape}")
        print(f"Terrain data columns: {list(terrain_data.columns)}")
        
        print("\nPreparing terrain data for machine learning...")
        ml_ready = prepare_for_ml(terrain_data)
        print(f"ML-ready terrain data shape: {ml_ready.shape}")
        print(f"ML-ready columns: {list(ml_ready.columns)[:10]}...")
        
        # Try to load sediment data but don't fail if it doesn't exist
        sediment_data = None
        try:
            print(f"\nTrying to load sediment data from: {sediment_file}")
            sediment_data = load_sediment_data(sediment_file)
            print(f"Loaded sediment data shape: {sediment_data.shape}")
            
            print("\nTrying to join sediment and terrain data...")
            combined_data = load_and_prepare_data(sediment_file, terrain_file)
            print(f"Combined data shape: {combined_data.shape}")
            print("Successfully joined sediment and terrain data!")
        except FileNotFoundError:
            print(f"Sediment data file not found. Using only terrain data for demonstration.")
        except Exception as e:
            print(f"Error working with sediment data: {str(e)}")
        
        # Create visualization
        print("\nCreating pydeck visualization...")
        if sediment_data is not None:
            deck = visualize_combined_data(terrain_data, sediment_data)
            print("Created visualization with terrain and sediment data")
        else:
            deck = visualize_terrain_data(terrain_data)
            print("Created visualization with terrain data only")
        
        # Render visualization in notebook or save as HTML
        try:
            # If in notebook, this will display the visualization
            display(deck)
        except NameError:
            # If not in notebook, save as HTML
            html_path = os.path.join(os.path.dirname(get_data_dir()), "visualizations")
            os.makedirs(html_path, exist_ok=True)
            html_file = os.path.join(html_path, "terrain_visualization.html")
            deck.to_html(html_file)
            print(f"Saved visualization to {html_file}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        print("Please ensure the terrain data file exists in the data directory.")

if __name__ == "__main__":
    main()
