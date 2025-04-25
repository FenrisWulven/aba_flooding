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
def visualize_terrain(file_path):
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
    
    deck = pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9',
        tooltip={"text": "Properties: {properties}"}
    )
    
    return deck


def create_dynamic_heatmap(file_path, timesteps=10):
    """Create a terrain visualization with time-based heatmap controlled by slider."""
    gdf = load_terrain_data(file_path)
    
    # Calculate center point for view
    original_crs = gdf.crs
    centroids = gdf.geometry.centroid
    center_x, center_y = centroids.x.mean(), centroids.y.mean()
    center_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([center_x], [center_y]), crs=original_crs)
    center_point_wgs84 = center_point.to_crs(epsg=4326)
    center_lon, center_lat = center_point_wgs84.geometry.x[0], center_point_wgs84.geometry.y[0]
    
    # Convert to WGS84 for PyDeck
    gdf = gdf.to_crs(epsg=4326)
    pydeck_data = gdf_to_pydeck(gdf)
    
    # Create multiple layers for different time steps
    layers = []
    for i in range(timesteps):
        # Simulate different values at different time steps
        # In a real implementation, you would use actual data for each time step
        color_value = i / timesteps
        
        # Example: colors transition from blue (cool) to red (hot)
        r = int(255 * color_value)
        b = int(255 * (1 - color_value))
        g = int(100 * min(color_value, 1-color_value))
        
        layer = pdk.Layer(
            "GeoJsonLayer",
            id=f"heatmap-layer-{i}",
            data=pydeck_data,
            opacity=0.8,
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="properties.MINKOTE * 20" if "MINKOTE" in gdf.columns else 10,
            get_fill_color=[r, g, b, 150],
            get_line_color=[r, g, b],
            get_line_width=1,
            pickable=True,
            visible=False,  # All layers start hidden
        )
        layers.append(layer)
    
    # Set the first layer to be visible initially
    layers[0].visible = True
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12,
        pitch=45,
        bearing=0
    )
    
    deck = pdk.Deck(
        layers=layers, 
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9',
        tooltip={"text": "Properties: {properties}"}
    )
    
    # Generate HTML with interactive slider
    slider_html = """
    <html>
    <head>
        <title>Flooding Simulation</title>
        <style>
            #slider-container {
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                width: 80%;
                text-align: center;
                background: rgba(255,255,255,0.8);
                padding: 10px;
                border-radius: 5px;
                z-index: 1000;
            }
            #slider {
                width: 100%;
            }
            .slider-label {
                font-weight: bold;
                margin-bottom: 5px;
            }
        </style>
    </head>
    <body>
        <div id="deck-container">
            {deck_html}
        </div>
        <div id="slider-container">
            <div class="slider-label">Time Step: <span id="slider-value">0</span></div>
            <input type="range" id="slider" min="0" max="{max_value}" value="0" step="1">
        </div>
        <script>
            const slider = document.getElementById('slider');
            const sliderValue = document.getElementById('slider-value');
            const totalLayers = {total_layers};
            
            // Function to update visible layer based on slider
            function updateVisibleLayer(value) {{
                // Hide all layers
                for (let i = 0; i < totalLayers; i++) {{
                    try {{
                        const deckLayer = document.querySelector(`[id$="heatmap-layer-${{i}}"]`);
                        if (deckLayer) {{
                            deckLayer.style.display = 'none';
                        }}
                    }} catch (e) {{
                        console.error(`Error updating layer ${{i}}:`, e);
                    }}
                }}
                
                // Show only the layer corresponding to current value
                try {{
                    const currentLayer = document.querySelector(`[id$="heatmap-layer-${{value}}"]`);
                    if (currentLayer) {{
                        currentLayer.style.display = 'block';
                    }}
                }} catch (e) {{
                    console.error(`Error showing layer ${{value}}:`, e);
                }}
                
                // Update display value
                sliderValue.textContent = value;
            }}
            
            // Initialize with first layer visible
            updateVisibleLayer(0);
            
            // Add event listener for slider changes
            slider.addEventListener('input', function() {{
                updateVisibleLayer(parseInt(this.value));
            }});
        </script>
    </body>
    </html>
    """
    
    # Generate the PyDeck HTML and combine with slider
    deck_html = deck.to_html(as_string=True, notebook_display=False)
    final_html = slider_html.format(
        deck_html=deck_html, 
        max_value=timesteps-1,
        total_layers=timesteps
    )
    
    # Save the combined HTML
    output_file = "flooding_simulation.html"
    with open(output_file, "w") as f:
        f.write(final_html)
    
    print(f"Dynamic heatmap visualization saved to {output_file}")
    return output_file


# Run visualization
if __name__ == "__main__":
    terrain_file = "Sediment.geojson"
    deck = visualize_terrain(terrain_file)
    
    # Save to HTML
    html_file = "terrain_visualization.html"
    deck.to_html(html_file)
    print(f"Visualization saved to {html_file}")

    html_file = create_dynamic_heatmap(terrain_file, timesteps=20)

    # Try to show in browser
    import webbrowser
    webbrowser.open(html_file)
    
    # # For Jupyter notebooks
    # try:
    #     display(deck)
    # except NameError:
    #     print("Not in a notebook environment. View the saved HTML file instead.")

