import bokeh
import geopandas as gpd
import os
from pathlib import Path
import json
import pandas as pd

from geo_utils import survival_layer

from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource, HoverTool, CheckboxGroup, CustomJS
from bokeh.models import WMTSTileSource, Column, Slider
from bokeh.transform import linear_cmap
from bokeh.layouts import column
from pyproj import Transformer
from shapely.ops import transform

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

def init_map():
    """Initialize a Bokeh map with terrain data, sediment layers, and flood risk predictions."""
    # Load the terrain data
    print("Loading terrain data...")
    terrain_data = load_terrain_data("Terrain.geojson")
    print(f"Loaded terrain data: {len(terrain_data)} features")
    print(f"Sample terrain geometry: {terrain_data.geometry.iloc[0] if len(terrain_data) > 0 else 'No data'}")
    
    # Check CRS and reproject if needed 
    if terrain_data.crs is None:
        print("Warning: GeoDataFrame has no CRS defined. Assuming EPSG:25832 (UTM Zone 32N / ETRS89)")
        terrain_data.set_crs(epsg=25832, inplace=True)
    
    # Convert to Web Mercator for Bokeh
    if terrain_data.crs != "EPSG:3857":
        print("Converting to Web Mercator...")
        terrain_mercator = terrain_data.to_crs(epsg=3857)
    else:
        terrain_mercator = terrain_data
        
    # Load sediment data
    print("Loading sediment data...")
    try:
        sediment_data = load_terrain_data("Sediment.geojson")
        
        # Check CRS and reproject if needed
        if sediment_data.crs is None:
            sediment_data.set_crs(epsg=25832, inplace=True)
            
        # Convert to Web Mercator
        if sediment_data.crs != "EPSG:3857":
            sediment_mercator = sediment_data.to_crs(epsg=3857)
        else:
            sediment_mercator = sediment_data
            
        # Convert to GeoJSON
        sediment_geojson = gdf_to_geojson(sediment_mercator)
        sediment_source = GeoJSONDataSource(geojson=sediment_geojson)
        
        has_sediment_data = True
    except Exception as e:
        print(f"Could not load sediment data: {e}")
        has_sediment_data = False
    
    # Convert terrain to GeoJSON for Bokeh GeoJSONDataSource
    terrain_geojson = gdf_to_geojson(terrain_mercator)
    terrain_source = GeoJSONDataSource(geojson=terrain_geojson)
    
    # Calculate the bounds for better centering
    denmark_bounds_x = (665000, 1550000)  # West to East
    denmark_bounds_y = (7000000, 8175000)  # South to North
    
    # Create figure with appropriate bounds
    p = figure(title="Flooding Risk Map", 
               x_axis_type="mercator", y_axis_type="mercator",
               x_range=denmark_bounds_x, y_range=denmark_bounds_y,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               width=1200, height=900)
    
    # Add hover tool for flood probabilities
    hover = HoverTool(tooltips=[
        ("Flood Probability", "@flood_probability{0.00%}"),
        ("Elevation", "@elevation"),
        ("jordart", "@jordart")
    ])
    p.add_tools(hover)
    
    # Add a base map
    cartodb_positron = WMTSTileSource(
        url='https://tiles.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        attribution='© OpenStreetMap contributors, © CartoDB'
    )
    p.add_tile(cartodb_positron)
    
    # Add terrain data layer
    terrain_layer = p.patches('xs', 'ys', source=terrain_source,
                            fill_color='green', fill_alpha=0.3,
                            line_color='black', line_width=0.2,
                            legend_label="Terrain")
    
    # Add sediment layer if available
    if has_sediment_data:
        sediment_layer = p.patches('xs', 'ys', source=sediment_source,
                                  fill_color='brown', fill_alpha=0.4,
                                  line_color='black', line_width=0.2,
                                  legend_label="Sediment")
    
    # Add slider for year control
    year_slider = Slider(start=0, end=10, value=0, step=1, title="Years into future")
    
    # Prepare layer control checkbox
    layer_names = ["Terrain", "Sediment"]
    active_layers = [0,1]  # Both terrain and sediment are off by default
    
    # Process flood risk layer
    flood_layer = None
    if has_sediment_data:
        print("Creating flood risk heatmap...")
        try:
            # Get initial flood risk data for year 0
            flood_source = survival_layer(
                terrain_geojson, 
                sediment_geojson, 
                year=0, 
                model=None,
                terrain_df=terrain_mercator,  # Pass the already loaded dataframes
                sediment_df=sediment_mercator
            )
            
            # Debug the GeoJSONDataSource
            if flood_source and flood_source.geojson:
                # Parse the geojson to verify it has features
                geojson_data = json.loads(flood_source.geojson)
                has_features = len(geojson_data.get('features', [])) > 0
                print(f"Source has features: {has_features}")
                
                if not has_features:
                    print("Warning: GeoJSON source doesn't contain any features")
                    raise ValueError("Empty GeoJSON data for patches renderer")
            else:
                print("Warning: Invalid GeoJSON source")
                raise ValueError("Invalid GeoJSON source for patches renderer")
            
            # Create color mapper for flood probabilities
            color_mapper = linear_cmap(
                field_name='flood_probability', 
                palette=Viridis256, 
                low=0, 
                high=1
            )
            
            # Add flood risk layer
            flood_layer = p.patches(
                'xs', 'ys', 
                source=flood_source,
                fill_color=color_mapper,
                fill_alpha=0.7,
                line_color=None,  # No borders for cleaner visualization
                legend_label="Flood Risk"
            )
                    
            # Add flood risk to layer controls
            layer_names.append("Flood Risk")
            active_layers.append(2)  # Show flood risk by default
            
            # Add color bar for flood risk
            from bokeh.models import ColorBar
            color_bar = ColorBar(
                color_mapper=color_mapper['transform'],
                location=(0, 0),
                title="Flood Risk",
                ticker=bokeh.models.BasicTicker(desired_num_ticks=5),
                formatter=bokeh.models.PrintfTickFormatter(format="%.0f%%")
            )
            p.add_layout(color_bar, 'right')
            
            # Create callback for year slider to update flood layer
            year_callback = CustomJS(args=dict(
                flood_layer=flood_layer,
                slider=year_slider
            ), code="""
                // Get the current year value
                const year = slider.value;
                
                // Adjust the color mapper based on the year
                // As year increases, the display intensifies
                flood_layer.glyph.fill_alpha = Math.min(0.6 + year * 0.04, 0.9);
                
                // For a fully dynamic prediction model, you would need server-side Bokeh
            """)
            year_slider.js_on_change('value', year_callback)
        except Exception as e:
            print(f"Error creating flood prediction layer: {e}")
            import traceback
            traceback.print_exc()
            flood_layer = None
    
    # Create checkbox for layer visibility
    checkbox = CheckboxGroup(labels=layer_names, active=active_layers)
    
    # Set up callback for toggling layers
    if has_sediment_data and flood_layer is not None:
        callback = CustomJS(args=dict(
            terrain_layer=terrain_layer,
            sediment_layer=sediment_layer,
            flood_layer=flood_layer,
            checkbox=checkbox
        ), code="""
            terrain_layer.visible = checkbox.active.includes(0);
            sediment_layer.visible = checkbox.active.includes(1);
            flood_layer.visible = checkbox.active.includes(2);
        """)
    elif has_sediment_data:
        callback = CustomJS(args=dict(
            terrain_layer=terrain_layer,
            sediment_layer=sediment_layer,
            checkbox=checkbox
        ), code="""
            terrain_layer.visible = checkbox.active.includes(0);
            sediment_layer.visible = checkbox.active.includes(1);
        """)
    else:
        callback = CustomJS(args=dict(
            terrain_layer=terrain_layer,
            checkbox=checkbox
        ), code="""
            terrain_layer.visible = checkbox.active.includes(0);
        """)
    
    checkbox.js_on_change('active', callback)
    
    # Create layout with map and controls
    from bokeh.layouts import row
    controls = column(checkbox, year_slider)
    layout = row(controls, p)
    
    # Configure legend location and click policy
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    return layout

if __name__=="__main__":
    p = init_map()
    # Save to an HTML file and display in browser
    output_file("terrain_map.html")
    show(p)