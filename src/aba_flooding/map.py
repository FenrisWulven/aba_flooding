import bokeh
import os
import geopandas as gpd

from aba_flooding.geo_utils import load_terrain_data, gdf_to_geojson, wgs84_to_web_mercator, load_geojson, load_gpkg, load_terrain_data, gdf_to_geojson
from aba_flooding.model import FloodModel

from bokeh.models import ColorBar

from bokeh.palettes import Viridis256, Category10
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource, HoverTool, CheckboxGroup, CustomJS
from bokeh.models import WMTSTileSource, Column, Slider
from bokeh.transform import linear_cmap
from bokeh.layouts import column
from shapely.geometry import Polygon


MODEL_PATH = "models/"
MODEL_NAME = "flood_model.joblib"

def load_models(model_path):
    """Load a trained FloodModel from file, including split station files if available."""
    try:
        print(f"Loading model from {model_path}...")
        # Create a FloodModel instance first, then call load as instance method
        model = FloodModel()
        model.load(path=model_path)
        
        # Check if we need to load split station models
        if len(model.models) == 0 and len(model.stations) > 0:
            print(f"Main model has stations but no models. Looking for split station files...")
            
            # Check for a stations directory in the same location as the model file
            model_dir = os.path.dirname(model_path)
            stations_dir = os.path.join(model_dir, "flood_model_stations")
            
            # Also check for the _stations directory format used in train.py
            if not os.path.exists(stations_dir):
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                stations_dir = os.path.join(model_dir, f"{base_name}_stations")
            
            if os.path.exists(stations_dir):
                print(f"Found stations directory: {stations_dir}")
                loaded_count = 0
                
                # Load each station file
                for station in model.stations:
                    # Try both naming patterns
                    station_files = [
                        os.path.join(stations_dir, f"{station}.joblib"),   # Original pattern
                        os.path.join(stations_dir, f"station_{station}.joblib")  # Pattern from train.py
                    ]
                    
                    station_file = None
                    for potential_file in station_files:
                        if os.path.exists(potential_file):
                            station_file = potential_file
                            break
                    
                    if station_file:
                        try:
                            print(f"Loading station model from: {station_file}")
                            station_models = model.load_station(station, os.path.dirname(station_file))
                            loaded_count += len(station_models) if station_models else 0
                        except Exception as e:
                            print(f"Error loading station {station}: {e}")
                    else:
                        print(f"No file found for station {station}, tried patterns: {station_files}")
                
                print(f"Loaded {loaded_count} models for {len(model.stations)} stations")
            else:
                print(f"No stations directory found at {stations_dir}")
        
        print(f"Model loaded successfully with {len(model.models)} models across {len(model.stations)} stations")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def repair_geometries(gdf):
    """Repair invalid geometries in a GeoDataFrame"""
    invalid_count = sum(~gdf.geometry.is_valid)
    if invalid_count > 0:
        print(f"Fixing {invalid_count} invalid geometries")
        # buffer(0) is a common trick to fix many geometry issues
        gdf.geometry = gdf.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
        still_invalid = sum(~gdf.geometry.is_valid)
        if still_invalid > 0:
            print(f"Warning: {still_invalid} geometries still invalid after repair")
    return gdf

def init_map():
    """Initialize a Bokeh map with terrain data, sediment layers, and flood risk predictions using FloodModel."""

    # Load sediment data
    print("Loading sediment data...")
    try:
        sediment_data = load_terrain_data("Sediment_wgs84.geojson")
        print(f"Loaded sediment data with CRS: {sediment_data.crs}")
        
        sjaelland_polygon = Polygon([
            (10.8, 54.9),   # Southwest corner
            (10.8, 56.1),   # Northwest corner
            (12.6, 56.1),   # Northeast corner
            (12.6, 54.9),   # Southeast corner
            (10.8, 54.9)    # Close the polygon
        ])
        
        # Make sure sediment data has correct CRS before filtering
        if sediment_data.crs is None:
            sediment_data.crs = "EPSG:4326"
        elif str(sediment_data.crs).upper() != "EPSG:4326":
            # Convert to WGS84 for filtering
            filter_data = sediment_data.to_crs("EPSG:4326")
        else:
            filter_data = sediment_data
           # Create GeoDataFrame for the filter polygon
        filter_gdf = gpd.GeoDataFrame(geometry=[sjaelland_polygon], crs="EPSG:4326")
        
        # Spatial join to filter sediment data
        initial_count = len(filter_data)
        filter_data = gpd.sjoin(filter_data, filter_gdf, predicate='within')
        print(f"Filtered from {initial_count} to {len(filter_data)} features (Sjælland only)")
        
        # If no features remain, it might mean polygon needs adjustment or different approach
        if len(filter_data) == 0:
            print("Warning: No features within Sjælland bounds, using intersects instead of within")
            filter_data = gpd.sjoin(sediment_data.to_crs("EPSG:4326"), filter_gdf, predicate='intersects')
            print(f"Intersects filtering resulted in {len(filter_data)} features")
        
        # Update sediment_data with filtered version
        sediment_data = filter_data
        
        # Now proceed with CRS conversion for display
        target_crs = "EPSG:3857"
        if str(sediment_data.crs).upper() != target_crs:
            print(f"Converting filtered sediment data from {sediment_data.crs} to {target_crs}")
            sediment_data = repair_geometries(sediment_data)
            sediment_mercator = sediment_data.to_crs(target_crs)
        else:
            sediment_mercator = sediment_data
            
        has_sediment_data = True
    except Exception as e:
        print(f"Could not load or filter sediment data: {e}")
        import traceback
        traceback.print_exc()
        has_sediment_data = False
        return None
    
    
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
    
    # Load coverage data
    try:
        print("Loading precipitation coverage data...")
        coverage_data = load_geojson("precipitation_coverage.geojson")
        
        if coverage_data is not None and not coverage_data.empty:
            # Ensure data is in Web Mercator projection
            if coverage_data.crs != "EPSG:3857":
                coverage_mercator = coverage_data.to_crs(epsg=3857)
            else:
                coverage_mercator = coverage_data
            
            # Convert to GeoJSON for Bokeh
            coverage_geojson = gdf_to_geojson(coverage_mercator)
            coverage_source = GeoJSONDataSource(geojson=coverage_geojson)
            
            # Add precipitation coverage polygons
            precipitation_layer = p.patches(
                'xs', 'ys',
                source=coverage_source,
                fill_color='blue',
                fill_alpha=0.2,
                line_color='blue',
                line_width=1,
                legend_label="Precipitation Coverage"
            )
            
            # Create a hover tool for precipitation areas
            precip_hover = HoverTool(
                tooltips=[
                    ("Station ID", "@station_id"),
                    ("Avg Precipitation", "@avg_precipitation{0.0} mm")
                ],
                renderers=[precipitation_layer]
            )
            p.add_tools(precip_hover)
            
            # Extract station points (centroids of coverage areas) for visualization
            stations_gdf = gpd.GeoDataFrame(
                coverage_mercator.copy(),
                geometry=coverage_mercator.geometry.centroid,
                crs=coverage_mercator.crs
            )
            
            # Convert station points to GeoJSON
            stations_geojson = gdf_to_geojson(stations_gdf)
            stations_source = GeoJSONDataSource(geojson=stations_geojson)
            
            # Add station points
            station_layer = p.circle(
                'x', 'y',
                source=stations_source,
                size=8,
                color='blue',
                fill_alpha=1.0,
                line_color='white',
                line_width=1
            )
            
            print(f"Successfully loaded precipitation coverage with {len(coverage_mercator)} areas")
        else:
            print("No precipitation coverage data found or it's empty")
    except Exception as e:
        print(f"Failed to load precipitation coverage: {e}")
        import traceback
        traceback.print_exc()
        
    # Add sediment layer if available
    print("before sediment layer")
    # sediment_layer = None
    # if has_sediment_data:
    #     sediment_geojson = gdf_to_geojson(sediment_mercator)
    #     sediment_source = GeoJSONDataSource(geojson=sediment_geojson)
    #     sediment_layer = p.patches('xs', 'ys', source=sediment_source,
    #                             fill_color='brown', fill_alpha=0.4,
    #                             line_color='black', line_width=0.2,
    #                             legend_label="Sediment")

    # Add sediment layer if available
    sediment_layer = None
    if has_sediment_data:
        try:
            print(f"Preparing sediment layer with {len(sediment_mercator)} features...")
            
            # First ensure all geometries are valid
            sediment_mercator = repair_geometries(sediment_mercator)
            
            # Simplify complex geometries if needed
            # if len(sediment_mercator) > 1000:
            #     print("Large number of sediment features detected, simplifying...")
            #     sediment_mercator = sediment_mercator.copy()
            #     sediment_mercator.geometry = sediment_mercator.geometry.simplify(1.0)
            
            # Check for required fields for patches
            if not all(col in sediment_mercator.columns for col in ['xs', 'ys']):
                print("Converting geometries to xs/ys coordinates for Bokeh...")
                
                # Custom function to extract coordinates from geometries
                def extract_polygon_coords(gdf):
                    """Extract xs and ys coordinates from polygon geometries"""
                    xs_list = []
                    ys_list = []
                    
                    for geom in gdf.geometry:
                        if geom.geom_type == 'Polygon':
                            # Extract exterior coordinates
                            xs, ys = geom.exterior.xy
                            xs_list.append(list(xs))
                            ys_list.append(list(ys))
                        elif geom.geom_type == 'MultiPolygon':
                            # For multipolygons, extract each polygon's coordinates
                            multi_xs = []
                            multi_ys = []
                            for poly in geom.geoms:
                                xs, ys = poly.exterior.xy
                                multi_xs.append(list(xs))
                                multi_ys.append(list(ys))
                            xs_list.append(multi_xs)
                            ys_list.append(multi_ys)
                    
                    gdf = gdf.copy()
                    gdf['xs'] = xs_list
                    gdf['ys'] = ys_list
                    return gdf
                
                # Try explode to handle MultiPolygons
                if 'MultiPolygon' in sediment_mercator.geometry.geom_type.values:
                    print("Exploding MultiPolygons...")
                    sediment_mercator = sediment_mercator.explode(index_parts=False)
                    print(f"Exploded MultiPolygons, now have {len(sediment_mercator)} features")
                
                # Add xs and ys coordinates
                sediment_mercator = extract_polygon_coords(sediment_mercator)
                
                # Convert to GeoJSON with safer method
                sediment_geojson = gdf_to_geojson(sediment_mercator)
                sediment_source = GeoJSONDataSource(geojson=sediment_geojson)
            else:
                # Direct method if xs/ys already exist
                sediment_geojson = gdf_to_geojson(sediment_mercator)
                sediment_source = GeoJSONDataSource(geojson=sediment_geojson)
            
            # Now create the layer with better error handling
            sediment_layer = p.patches('xs', 'ys', source=sediment_source,
                                fill_color='brown', fill_alpha=0.4,
                                line_color='black', line_width=0.2,
                                legend_label="Sediment")
            
            print("Successfully added sediment layer")
        except Exception as e:
            print(f"Failed to add sediment layer: {e}")
            import traceback
            traceback.print_exc()
            has_sediment_data = False  # Mark as unavailable since we couldn't use it

    print("after sediment layer")
    # Prepare FloodModel predictions
    flood_layer = None
    year_slider = Slider(start=0, end=10, value=0, step=1, title="Years into future")
    if has_sediment_data:
        try:            
            # Train FloodModel with all soil types from sediment data
            if MODEL_NAME in os.listdir(MODEL_PATH):
                flood_model = load_models(MODEL_PATH + MODEL_NAME)
            else:
                print("Training new flood models...")
                # Train models for all soil types
                flood_model = train_all_models(soil_types, stationId)
                # Plot models for available soil types
                #flood_model.plot_all(save=True)

            print("model?")
            # Precompute predictions for all years
            sediment_with_predictions = sediment_mercator.copy()
            for year in range(0, 2):
                sediment_with_predictions = flood_model.predict_proba(sediment_with_predictions, station_coverage=stations_gdf, year=year)
                print("Predicted Year ",year)
            # Convert to GeoJSON data source
            print("Predicted all years")
            flood_geojson = gdf_to_geojson(sediment_with_predictions)
            flood_source = GeoJSONDataSource(geojson=flood_geojson)
            print("converted to geojson")
            # Create color mapper with initial field name for year 0
            color_mapper = linear_cmap(
                field_name='predictions_0',
                palette=Viridis256,
                low=0,
                high=100  # Predictions are percentages (0-100)
            )
            
            print("Create Flood Layer")
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
                    flood_layer=flood_layer,
                    flood_source=flood_source,
                    slider=year_slider,
                    color_bar=color_bar,
                    mapper=color_mapper
                ),
                code="""
                    // Ensure we're in a browser context with a DOM
                    if (typeof document === 'undefined' || document.body === null) {
                        console.warn('DOM not ready yet, skipping callback');
                        return;
                    }
                    
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
            
            print("Create hover")
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
