import geopandas as gpd
import pydeck as pdk
from shapely.geometry import box
import os
from pathlib import Path
import pandas as pd

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

# Filter for Copenhagen (approximate bounding box for central Copenhagen)
# def filter_copenhagen(gdf):
#     copenhagen_bbox = box(
#     return gdf[gdf.intersects(copenhagen_bbox)]

# Convert GeoDataFrame to Pydeck-compatible format
def gdf_to_pydeck(gdf):
    """Convert a GeoDataFrame to PyDeck format, ensuring WGS84 projection."""
    # Make a copy to avoid modifying the original
    gdf = gdf.copy()
    
    # Check CRS and reproject to WGS84 if needed
    if gdf.crs is None:
        print("Warning: GeoDataFrame has no CRS defined. Assuming EPSG:25832 (UTM Zone 32N / ETRS89)")
        gdf.set_crs(epsg=25832, inplace=True)
    
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting from {gdf.crs} to WGS84 (EPSG:4326)")
        gdf = gdf.to_crs(epsg=4326)
    
    # Extract properties in addition to geometry
    features = []
    for idx, row in gdf.iterrows():
        properties = {col: row[col] for col in gdf.columns if col != 'geometry'}
        
        # Handle NaN values which aren't JSON serializable
        for key, value in properties.items():
            if pd.isna(value):
                properties[key] = None
                
        feature = {
            "type": "Feature", 
            "geometry": row.geometry.__geo_interface__, 
            "properties": properties
        }
        features.append(feature)
    
    return {"type": "FeatureCollection", "features": features}

# Visualize using Pydeck
def visualize_terrain(file_path, output_dir=None):
    gdf = load_terrain_data(file_path)
    print(f"Data loaded from {file_path}")
    print(f"Number of features: {len(gdf)}")
    
    print("columns:", gdf.columns)
    print("geometry type sample:", gdf.geometry.iloc[0].geom_type)
    print(f"Coordinate reference system: {gdf.crs}")
    
    # Calculate center coordinates correctly before reprojection
    # Store the original CRS
    original_crs = gdf.crs
    
    # Calculate centroids in the original projected CRS
    centroids = gdf.geometry.centroid
    center_x = centroids.x.mean()
    center_y = centroids.y.mean()
    
    # Create a single point for the center and convert to WGS84
    center_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([center_x], [center_y]), crs=original_crs)
    center_point_wgs84 = center_point.to_crs(epsg=4326)
    center_lon = center_point_wgs84.geometry.x[0]
    center_lat = center_point_wgs84.geometry.y[0]
    
    # Calculate the bounding box of the data to determine appropriate zoom
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
    
    # Print bounds for debugging
    print(f"Data bounds (WGS84): {bounds}")
    
    # Now prepare the data for PyDeck
    pydeck_data = gdf_to_pydeck(gdf)
    
    elevation_property = None
    # Check for potential elevation columns
    for col in ['MINKOTE', 'MAXKOTE', 'elevation', 'height', 'z']:
        if col in gdf.columns:
            elevation_property = f'properties.{col}'
            break
    
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=pydeck_data,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=True,  # Enable 3D
        wireframe=True,
        get_elevation=elevation_property or 10,  # Use identified elevation or default
        elevation_scale=20,  # Reduced scale to see more area
        get_fill_color=[255, 165, 0, 100],  # Orange color with transparency
        get_line_color=[255, 100, 0],
        get_line_width=1,
        pickable=True,  # Enable tooltips
    )
    
    # Adjust zoom level to show all of Denmark (lower number = more zoomed out)
    # Denmark is approximately 300km x 350km
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=6,  # Zoomed out to see all of Denmark
        pitch=30,  # Reduced pitch for better overview
        bearing=0
    )
    
    # Set output directory for HTML
    if output_dir is None:
        # Default to the docs/visualizations directory in the project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        output_dir = project_root / "docs" / "visualizations"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the deck
    deck = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9',
        tooltip={"text": "Properties: {properties}"}
    )
    
    # Save as HTML in the output directory
    html_file = os.path.join(output_dir, "terrain_visualization.html")
    deck.to_html(html_file, iframe_width="100%", iframe_height="600")
    print(f"Visualization saved to {html_file}")
    
    return deck

# Run visualization
if __name__ == "__main__":
    terrain_file = "Terrain.geojson"
    
    # Create docs/visualizations directory in project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    visualizations_dir = project_root / "docs" / "visualizations"
    
    deck = visualize_terrain(terrain_file, visualizations_dir)
    
    # Save to HTML in the docs/visualizations directory
    html_file = os.path.join(visualizations_dir, "terrain_visualization.html")
    
    # Print GitHub Pages URL
    print("\nYour visualization will be available at:")
    print(f"https://[your-username].github.io/aba_flooding/visualizations/terrain_visualization.html")
    
    # Try to show in browser
    import webbrowser
    webbrowser.open(html_file)

