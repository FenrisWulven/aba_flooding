import bokeh
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import geopandas as gpd

from geo_utils import load_terrain_data, gdf_to_geojson, wgs84_to_web_mercator, load_geojson, load_gpkg, load_terrain_data, gdf_to_geojson

from train import train_all_models, load_models
# Add import for preprocess module if needed
from preprocess import preprocess_data_for_survival, load_process_data

from bokeh.models import LinearColorMapper
from bokeh.models import ColorBar

from bokeh.palettes import Viridis256, Category10
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource, HoverTool, CheckboxGroup, CustomJS
from bokeh.models import WMTSTileSource, Column, Slider
from bokeh.transform import linear_cmap
from bokeh.layouts import column


MODEL_PATH = "models/"
MODEL_NAME = "flood_models.pkl"

def init_map():
    """Initialize a Bokeh map with terrain data, sediment layers, and flood risk predictions using FloodModel."""

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
        return
    
    
    # Prepare map figure
    denmark_bounds_x = (670000, 1500000)
    denmark_bounds_y = (7000000, 8170000)
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
    
    # Create precipitation coverage layer
    precipitation_layer = None
    station_layer = None
    
    if denmark is not None:
        # Create precipitation coverage areas
        print("\n=== Creating precipitation coverage areas ===")
        precip_coverage_gdf, stations_gdf = create_precipitation_coverage(denmark)
        
        if precip_coverage_gdf is not None and not precip_coverage_gdf.empty:
            print(f"Successfully created coverage GeoDataFrame with {len(precip_coverage_gdf)} polygons")
            # Create GeoJSON data source for precipitation coverage
            precip_geojson = gdf_to_geojson(precip_coverage_gdf)
            precip_source = GeoJSONDataSource(geojson=precip_geojson)
            
            # Create color mapper for precipitation values
            if 'avg_precipitation' in precip_coverage_gdf.columns:
                color_mapper = linear_cmap(
                    field_name='avg_precipitation',
                    palette=Category10[10],  # Use a different palette than flood risk
                    low=precip_coverage_gdf['avg_precipitation'].min(),
                    high=precip_coverage_gdf['avg_precipitation'].max()
                )
                
                # Add precipitation coverage layer
                precipitation_layer = p.patches(
                    'xs', 'ys',
                    source=precip_source,
                    fill_color=color_mapper,
                    fill_alpha=0.5,
                    line_color='black',
                    line_width=0.5,
                    legend_label="Precipitation Coverage"
                )
                
                # Add hover for precipitation data
                precip_hover = HoverTool(
                    tooltips=[
                        ("Station ID", "@station_id"),
                        ("Avg Precipitation", "@avg_precipitation{0.00} mm")
                    ],
                    renderers=[precipitation_layer]
                )
                p.add_tools(precip_hover)
            else:
                # Fallback if no precipitation data available
                precipitation_layer = p.patches(
                    'xs', 'ys',
                    source=precip_source,
                    fill_color='blue',
                    fill_alpha=0.3,
                    line_color='black',
                    line_width=0.5,
                    legend_label="Precipitation Coverage"
                )
            
            # Add station points if available
            if stations_gdf is not None and not stations_gdf.empty:
                try:
                    # Determine the station ID column from the stations GeoDataFrame
                    station_id_column = None
                    for possible_id in ['id', 'station_id', 'station', 'name', 'station_name', 'temp_id']:
                        if possible_id in stations_gdf.columns:
                            station_id_column = possible_id
                            break
                    
                    # Fallback to using the first column if no ID column is found
                    if station_id_column is None:
                        station_id_column = stations_gdf.columns[0]
                        print(f"No obvious ID column found, using '{station_id_column}' as station ID")
                    
                    # Convert DataFrame columns to serializable types
                    for col in stations_gdf.columns:
                        if col != 'geometry':
                            # Convert all columns to string to avoid serialization issues
                            stations_gdf[col] = stations_gdf[col].astype(str)
                    
                    # Create a simplified copy with only necessary columns
                    simple_stations = gpd.GeoDataFrame(
                        {'station_id': stations_gdf[station_id_column].astype(str)},
                        geometry=stations_gdf.geometry,
                        crs=stations_gdf.crs
                    )
                    
                    stations_geojson = gdf_to_geojson(simple_stations)
                    stations_source = GeoJSONDataSource(geojson=stations_geojson)
                    
                    station_layer = p.circle(
                        'x', 'y',
                        source=stations_source,
                        size=10,
                        color='red',
                        legend_label="Precipitation Stations"
                    )
                except Exception as station_error:
                    print(f"Error adding station points: {station_error}")
                    import traceback
                    traceback.print_exc()
                    station_layer = None
        else:
            print("ERROR: Failed to create precipitation coverage areas")

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
            for i, soil_type in enumerate(soil_types):
                # take the first two characters of the soil type
                soil_types[i] = soil_type.split(' ')[0]

            print(f"Found soil types in sediment data: {soil_types}")
            
            # Train FloodModel with all soil types from sediment data
            if MODEL_NAME in os.listdir(MODEL_PATH):
                flood_model = load_models(MODEL_PATH + MODEL_NAME)
            else:
                print("Training new flood models...")
                # Train models for all soil types
                flood_model = train_all_models(soil_types)
                # Plot models for available soil types
                #flood_model.plot_all(save=True)

            # Precompute predictions for all years
            sediment_with_predictions = sediment_mercator.copy()
            for year in range(0, 11):
                sediment_with_predictions = flood_model.predict_proba(sediment_with_predictions, year)

            # Convert to GeoJSON data source
            flood_geojson = gdf_to_geojson(sediment_with_predictions)
            flood_source = GeoJSONDataSource(geojson=flood_geojson)
            
            # Create color mapper with initial field name for year 0
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
                title="Flood Risk (%) - Year 0",
                ticker=bokeh.models.BasicTicker(desired_num_ticks=5),
                formatter=bokeh.models.PrintfTickFormatter(format="%d%%")
            )
            p.add_layout(color_bar, 'right')
            
            # Fixed slider callback to properly update the visualization
            year_callback = CustomJS(
                args=dict(
                    flood_layer=flood_layer,  # Pass the actual layer
                    flood_source=flood_source,
                    slider=year_slider,
                    color_bar=color_bar,
                    mapper=color_mapper  # Pass the entire mapper object
                ),
                code="""
                    // Get current year from slider
                    const year = Math.round(slider.value);
                    console.log('Changing visualization to year:', year);
                    
                    // Create the field name for this year's predictions
                    const field_name = 'predictions_' + year;
                    
                    // Need to update the mapper's field name
                    mapper.field = field_name;
                    
                    // Update the layer's glyph
                    flood_layer.glyph.fill_color = {
                        ...flood_layer.glyph.fill_color,
                        field: field_name
                    };
                    
                    // Update the color bar title
                    color_bar.title = 'Flood Risk (%) - Year ' + year;
                    
                    // Force a data source change to trigger redraw
                    flood_source.change.emit();
                """
            )
            year_slider.js_on_change('value', year_callback)
            
            # Create a single hover tool that will be dynamically updated
            hover = HoverTool(
                tooltips=[
                    ("Soil Type", "@sediment"),
                    ("Elevation", "@elevation{0,0.0}"),
                    ("Current Prediction (Year 0)", "@predictions_0{0.0}%")  # This should be updated by the slider
                ],
                renderers=[flood_layer]  # Explicitly attach to flood layer
            )
            
            # Improved tooltip callback that updates the hover display
            hover_callback = CustomJS(
                args=dict(hover=hover, slider=year_slider),
                code="""
                    // Get current year from slider
                    const year = Math.round(slider.value);
                    const field = 'predictions_' + year;
                    
                    // Update the hover tooltip with the current year
                    hover.tooltips[2][0] = "Current Prediction (Year " + year + ")";
                    hover.tooltips[2][1] = "@" + field + "{0.0}%";
                    
                    // Force the hover tool to update
                    hover.change.emit();
                    console.log("Updated hover tooltip for year: " + year);
                """
            )
            
            # Add the hover callback to the slider's change event
            year_slider.js_on_change('value', hover_callback)
            
            # Add the hover tool to the plot
            p.add_tools(hover)
            
        except Exception as e:
            print(f"Error setting up flood predictions: {e}")
            import traceback
            traceback.print_exc()
    
    # Layer visibility controls
    layer_names = []
    active_layers = []
    
    if sediment_layer:
        layer_names.append("Sediment")
        active_layers.append(len(layer_names) - 1)
    
    if precipitation_layer:
        layer_names.append("Precipitation Coverage")
        active_layers.append(len(layer_names) - 1)
    
    if flood_layer:
        layer_names.append("Flood Risk")
        active_layers.append(len(layer_names) - 1)
    
    checkbox = CheckboxGroup(labels=layer_names, active=active_layers)
    
    # JavaScript callback for layer visibility
    js_args = {'checkbox': checkbox}
    
    if sediment_layer:
        js_args['sediment_layer'] = sediment_layer
    
    if precipitation_layer:
        js_args['precipitation_layer'] = precipitation_layer
    
    if station_layer:
        js_args['station_layer'] = station_layer
    
    if flood_layer:
        js_args['flood_layer'] = flood_layer
    
    checkbox_code = """
        let i = 0;
    """
    
    if sediment_layer:
        checkbox_code += """
        sediment_layer.visible = checkbox.active.includes(i);
        i++;
        """
    
    if precipitation_layer:
        checkbox_code += """
        precipitation_layer.visible = checkbox.active.includes(i);
        if (typeof station_layer !== 'undefined')
            station_layer.visible = checkbox.active.includes(i);
        i++;
        """
    
    if flood_layer:
        checkbox_code += """
        if (typeof flood_layer !== 'undefined')
            flood_layer.visible = checkbox.active.includes(i);
        """
    
    checkbox_callback = CustomJS(args=js_args, code=checkbox_code)
    checkbox.js_on_change('active', checkbox_callback)
    
    # Assemble layout
    controls = column(year_slider, checkbox)
        
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