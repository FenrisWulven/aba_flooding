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


# QUick idea:
# Gets the data from the data folder and loads it into the map
# creates a checkbox for terrain and sediment.
# will create a slider that switches between layers of survival predictions (done in geo_utils)
# simple! voila! it doesnt have data for all of denmark for some god forsaken reason.

MODEL_PATH = "models/"
MODEL_NAME = "flood_models.pkl"

def voronoi_finite_polygons_2d(vor, radius=None):
    """Convert Voronoi diagram to finite polygons."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    
    new_regions = []
    new_vertices = vor.vertices.tolist()
    
    center = vor.points.mean(axis=0)
    # Fix for NumPy 2.0: Use np.ptp instead of .ptp() method
    radius = np.ptp(vor.points, axis=0).max() * 2 if radius is None else radius
    
    # Construct a map of all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        # Skip points that don't have any ridges (fixes KeyError)
        if p1 not in all_ridges:
            print(f"Skipping point {p1} which has no ridges")
            continue
            
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue
        
        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge
                continue
            
            # Infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices.mean(axis=0) + direction * radius
            
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        
        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        
        new_regions.append(new_region.tolist())
    
    return new_regions, np.asarray(new_vertices)

def create_precipitation_coverage(denmark_gdf):
    """Create Voronoi polygons for precipitation stations that cover Denmark without overlap."""
    try:
        print("\n=== Creating precipitation coverage areas ===")
        # Load precipitation data - stations are rows with longitude and latitude columns
        print("Loading precipitation data...")
        station_data = pd.read_parquet('data/raw/location.parquet')
        print(f"Loaded data with {len(station_data)} stations")
        
        # Determine the station ID column
        id_column = None
        for possible_id in ['id', 'station_id', 'station', 'name', 'station_name']:
            if possible_id in station_data.columns:
                id_column = possible_id
                break
        
        # If no obvious ID column is found, use the index
        if id_column is None:
            print("No obvious station ID column found, using index as station ID")
            station_data['temp_id'] = station_data.index.astype(str)
            id_column = 'temp_id'
        
        # Find longitude and latitude columns
        lon_col = next((col for col in station_data.columns if col.lower() in ['longitude', 'lon', 'long']), None)
        lat_col = next((col for col in station_data.columns if col.lower() in ['latitude', 'lat']), None)
        
        if lon_col is None or lat_col is None:
            raise ValueError(f"Could not identify longitude and latitude columns. Available columns: {station_data.columns.tolist()}")
        
        print(f"Using column '{lon_col}' for longitude, '{lat_col}' for latitude, and '{id_column}' for station IDs")
        
        # Create GeoDataFrame for stations
        print("Creating station GeoDataFrame...")
        stations_gdf = gpd.GeoDataFrame(
            station_data,
            geometry=gpd.points_from_xy(station_data[lon_col], station_data[lat_col]),
            crs="EPSG:4326"  # WGS84
        )
        
        # Reproject to match map projection
        print("Reprojecting to Web Mercator...")
        stations_gdf = stations_gdf.to_crs("EPSG:3857")  # Web Mercator
        
        # Print information about Denmark boundary
        print(f"Denmark GDF info: {denmark_gdf.shape}")
        print(f"Denmark CRS: {denmark_gdf.crs}")
        
        # Create Voronoi diagram
        print("Creating Voronoi diagram...")
        coords = np.array([(p.x, p.y) for p in stations_gdf.geometry])
        print(f"Number of station coordinates: {len(coords)}")
        
        # Check for duplicate or very close points
        _, unique_indices = np.unique(np.round(coords, decimals=5), axis=0, return_index=True)
        if len(unique_indices) < len(coords):
            print(f"Warning: Found {len(coords) - len(unique_indices)} potential duplicate points. Using only unique locations.")
            coords = coords[np.sort(unique_indices)]
            # Adjust stations_gdf to match unique points
            stations_gdf = stations_gdf.iloc[np.sort(unique_indices)].copy()
        
        # Get Denmark boundary for clipping
        boundary = denmark_gdf.geometry.union_all().bounds
        print(f"Denmark bounds: {boundary}")
        boundary_width = boundary[2] - boundary[0]
        boundary_height = boundary[3] - boundary[1]
        
        # Add corner points to ensure complete coverage
        corner_points = [
            [boundary[0] - boundary_width, boundary[1] - boundary_height],
            [boundary[2] + boundary_width, boundary[1] - boundary_height],
            [boundary[0] - boundary_width, boundary[3] + boundary_height],
            [boundary[2] + boundary_width, boundary[3] + boundary_height]
        ]
        
        all_points = np.vstack([coords, corner_points])
        print(f"Total points for Voronoi (including corners): {len(all_points)}")
        
        try:
            vor = Voronoi(all_points)
            print(f"Voronoi diagram created with {len(vor.points)} points and {len(vor.vertices)} vertices")
        except Exception as vor_error:
            print(f"Error creating Voronoi diagram: {vor_error}")
            # Add jitter to points to avoid collinearity issues
            jitter = np.random.normal(0, 0.00001, all_points.shape)
            all_points = all_points + jitter
            print("Added small jitter to points to avoid numerical issues, retrying...")
            vor = Voronoi(all_points)
        
        # Get Voronoi polygons
        print("Converting Voronoi diagram to polygons...")
        regions, vertices = voronoi_finite_polygons_2d(vor)
        print(f"Created {len(regions)} Voronoi regions")
        
        # Create clipped polygons for each station
        print("Creating clipped polygons...")
        voronoi_polygons = []
        valid_station_ids = []
        
        for i, region in enumerate(regions):
            if i < len(coords):  # Skip corner points
                try:
                    polygon = Polygon([vertices[v] for v in region])
                    # Clip polygon to Denmark boundary
                    clipped_polygon = polygon.intersection(denmark_gdf.geometry.union_all())
                    if not clipped_polygon.is_empty:
                        voronoi_polygons.append(clipped_polygon)
                        valid_station_ids.append(stations_gdf.iloc[i][id_column])
                except Exception as poly_error:
                    print(f"Error creating polygon for region {i}: {poly_error}")
                    continue
        
        print(f"Created {len(voronoi_polygons)} valid polygons")
        
        # Create GeoDataFrame with coverage areas
        coverage_gdf = gpd.GeoDataFrame(
            {'station_id': valid_station_ids},
            geometry=voronoi_polygons,
            crs=stations_gdf.crs
        )
        
        # Add precipitation data to coverage areas if available
        print("Adding station data to polygons...")
        precipitation_column = next((col for col in station_data.columns if any(x in col.lower() for x in ['precip', 'rain', 'rainfall'])), None)
        
        if precipitation_column:
            print(f"Found precipitation data in column: {precipitation_column}")
            for i, station_id in enumerate(valid_station_ids):
                try:
                    # Find the row for this station in the original data
                    station_row = station_data[station_data[id_column] == station_id]
                    if len(station_row) > 0:
                        # Get precipitation value and convert to numeric
                        precip_value = pd.to_numeric(station_row[precipitation_column].iloc[0], errors='coerce')
                        if not pd.isna(precip_value):
                            coverage_gdf.loc[i, 'avg_precipitation'] = precip_value
                        else:
                            print(f"Warning: Non-numeric precipitation value for station {station_id}. Using default.")
                            coverage_gdf.loc[i, 'avg_precipitation'] = 0
                except Exception as e:
                    print(f"Error processing precipitation data for station {station_id}: {e}")
        else:
            print("No precipitation data column found. Using random values for visualization.")
            # Generate random precipitation values for visualization
            coverage_gdf['avg_precipitation'] = np.random.uniform(0, 100, size=len(coverage_gdf))
        
        print(f"Successfully created coverage areas for {len(coverage_gdf)} stations")
        
        # Save to file for inspection
        try:
            coverage_gdf.to_file("precipitation_coverage.geojson", driver="GeoJSON")
            print("Saved coverage areas to precipitation_coverage.geojson")
        except Exception as e:
            print(f"Could not save coverage areas: {e}")
        
        return coverage_gdf, stations_gdf
    
    except Exception as e:
        print(f"Error creating precipitation coverage: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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
    
    # Load Denmark boundary for clipping
    try:
        # Instead of using deprecated geopandas.datasets.get_path
        # Use alternative approach to get Denmark boundary
        try:
            # Method 1: Try to use geopandas with URL to Natural Earth data
            import requests
            from io import BytesIO
            import zipfile
            
            # Download Natural Earth data
            url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Extract the zipfile
                z = zipfile.ZipFile(BytesIO(response.content))
                # Extract to a temporary location
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                z.extractall(temp_dir)
                
                # Find the shapefile
                shp_path = None
                for file in os.listdir(temp_dir):
                    if file.endswith(".shp"):
                        shp_path = os.path.join(temp_dir, file)
                        break
                
                if shp_path:
                    # Load and filter to Denmark
                    world = gpd.read_file(shp_path)
                    denmark = world[world.NAME == 'Denmark'].to_crs(epsg=3857)
                    
                    # Clean up temp directory
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
                else:
                    raise FileNotFoundError("Shapefile not found in the downloaded zip")
            else:
                raise ConnectionError(f"Failed to download data: {response.status_code}")
                
        except Exception as inner_e:
            print(f"Could not load Denmark boundary from Natural Earth: {inner_e}")
            
            # Method 2: Create a simplified Denmark boundary manually
            print("Creating simplified Denmark boundary...")
            # Approximate Denmark bounding box in EPSG:4326 (WGS84)
            # These coordinates represent a rough bounding box around Denmark
            denmark_coords = [
                (8.0, 54.5),   # Southwest corner
                (8.0, 57.8),   # Northwest corner
                (13.0, 57.8),  # Northeast corner
                (13.0, 54.5),  # Southeast corner
                (8.0, 54.5)    # Close the polygon
            ]
            
            # Create a polygon and convert to GeoDataFrame
            denmark_polygon = Polygon(denmark_coords)
            denmark = gpd.GeoDataFrame(
                {'name': ['Denmark']}, 
                geometry=[denmark_polygon], 
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            print("Using simplified Denmark boundary")
            
    except Exception as e:
        print(f"Could not load Denmark boundary: {e}")
        denmark = None
    
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