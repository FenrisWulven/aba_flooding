import bokeh
import pandas as pd

from geo_utils import load_terrain_data, gdf_to_geojson, wgs84_to_web_mercator, load_geojson, load_gpkg, load_terrain_data, gdf_to_geojson

from train import train_all_models

from bokeh.models import LinearColorMapper
from bokeh.models import ColorBar

from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource, HoverTool, CheckboxGroup, CustomJS
from bokeh.models import WMTSTileSource, Column, Slider
from bokeh.transform import linear_cmap
from bokeh.layouts import column


# QUick idea:
# Gets the data from the data folder and loads it into the map
# creates a checkbox for terrain and sediment.
# will create a slider that switches between layers of survival predictions (done in geo_utils)
# simple! voila! it doesnt have data for all of denmark for some god forsaken reason.


def init_map():
    """Initialize a Bokeh map with terrain data, sediment layers, and flood risk predictions using FloodModel."""
    # Load the terrain data
    print("Loading terrain data...")
    terrain_data = load_terrain_data("Terrain.geojson")
    
    # Check CRS and reproject to Web Mercator
    if terrain_data.crs != "EPSG:3857":
        terrain_mercator = terrain_data.to_crs(epsg=3857)
    else:
        terrain_mercator = terrain_data
        
    # Load sediment data
    print("Loading sediment data...")
    try:
        sediment_data = load_terrain_data("Sediment.geojson")
        if sediment_data.crs != "EPSG:3857":
            sediment_mercator = sediment_data.to_crs(epsg=3857)
        else:
            sediment_mercator = sediment_data
        has_sediment_data = True
    except Exception as e:
        print(f"Could not load sediment data: {e}")
        has_sediment_data = False
    
    # Prepare map figure
    denmark_bounds_x = (665000, 1550000)
    denmark_bounds_y = (7000000, 8175000)
    p = figure(title="Flood Risk Map", 
               x_axis_type="mercator", y_axis_type="mercator",
               x_range=denmark_bounds_x, y_range=denmark_bounds_y,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               width=1200, height=900)
    
    # Add base map tiles
    cartodb_positron = WMTSTileSource(
        url='https://tiles.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        attribution='© OpenStreetMap contributors, © CartoDB'
    )
    p.add_tile(cartodb_positron)
    
    # Add terrain layer
    terrain_geojson = gdf_to_geojson(terrain_mercator)
    terrain_source = GeoJSONDataSource(geojson=terrain_geojson)
    terrain_layer = p.patches('xs', 'ys', source=terrain_source,
                            fill_color='green', fill_alpha=0.3,
                            line_color='black', line_width=0.2,
                            legend_label="Terrain")
    
    # Add sediment layer if available
    sediment_layer = None
    if has_sediment_data:
        sediment_geojson = gdf_to_geojson(sediment_mercator)
        sediment_source = GeoJSONDataSource(geojson=sediment_geojson)
        sediment_layer = p.patches('xs', 'ys', source=sediment_source,
                                  fill_color='brown', fill_alpha=0.4,
                                  line_color='black', line_width=0.2,
                                  legend_label="Sediment")
    
    # Prepare FloodModel predictions
    flood_layer = None
    year_slider = Slider(start=0, end=10, value=0, step=1, title="Years into future")
    if has_sediment_data:
        try:
            # Extract soil types from sediment data
            soil_types = sediment_mercator['sediment'].unique().tolist()
            
            # Train FloodModel
            
            flood_model = train_all_models(soil_types)
            
            # Precompute predictions for all years
            sediment_with_predictions = sediment_mercator.copy()
            for year in range(0, 11):
                sediment_with_predictions = flood_model.predict_proba(sediment_with_predictions, year)
            
            # Convert to GeoJSON data source
            flood_geojson = gdf_to_geojson(sediment_with_predictions)
            flood_source = GeoJSONDataSource(geojson=flood_geojson)
            
            # Create color mapper
            color_mapper = linear_cmap(
                field_name='predictions_0',
                palette=Viridis256,
                low=0,
                high=100  # Predictions are percentages (0-100)
            )
            
            # Add flood risk layer
            flood_layer = p.patches(
                'xs', 'ys', 
                source=flood_source,
                fill_color=color_mapper,
                fill_alpha=0.7,
                line_color=None,
                legend_label="Flood Risk"
            )
            
            # Configure color bar
            color_bar = ColorBar(
                color_mapper=color_mapper['transform'],
                location=(0, 0),
                title="Flood Risk (%)",
                ticker=bokeh.models.BasicTicker(desired_num_ticks=5),
                formatter=bokeh.models.PrintfTickFormatter(format="%d%%")
            )
            p.add_layout(color_bar, 'right')
            
            # Set up slider callback to update color field
            year_callback = CustomJS(
                args=dict(transform=color_mapper['transform'], flood_source=flood_source),
                code="""
                    const year = cb_obj.value;
                    transform.field = 'predictions_' + year;
                    flood_source.change.emit();
                """
            )
            year_slider.js_on_change('value', year_callback)
            
        except Exception as e:
            print(f"Error setting up flood predictions: {e}")
            import traceback
            traceback.print_exc()
    
    # Configure hover tool
    hover = HoverTool(
        tooltips=[
            ("Flood Probability", "@predictions_0%"),
            ("Soil Type", "@jordart"),
            ("Elevation", "@elevation")
        ]
    )
    p.add_tools(hover)
    
    # Layer visibility controls
    layer_names = ["Terrain", "Sediment"]
    active_layers = [0, 1]
    if flood_layer:
        layer_names.append("Flood Risk")
        active_layers.append(2)
    
    checkbox = CheckboxGroup(labels=layer_names, active=active_layers)
    
    # JavaScript callback for layer visibility
    js_args = {'terrain_layer': terrain_layer, 'checkbox': checkbox}
    if sediment_layer:
        js_args['sediment_layer'] = sediment_layer
    if flood_layer:
        js_args['flood_layer'] = flood_layer
    
    checkbox_callback = CustomJS(args=js_args, code="""
        terrain_layer.visible = checkbox.active.includes(0);
        if (typeof sediment_layer !== 'undefined') 
            sediment_layer.visible = checkbox.active.includes(1);
        if (typeof flood_layer !== 'undefined') 
            flood_layer.visible = checkbox.active.includes(2);
    """)
    checkbox.js_on_change('active', checkbox_callback)
    
    # Assemble layout
    controls = column(checkbox, year_slider)
    layout = column(p, controls)
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.title = "Layers"
    
    return layout

if __name__=="__main__":
    p = init_map()
    # Save to an HTML file and display in browser
    output_file("terrain_map.html")
    show(p)