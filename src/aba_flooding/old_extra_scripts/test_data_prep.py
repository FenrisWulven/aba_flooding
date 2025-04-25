import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load location data - stations are rows with longitude and latitude columns
df = pd.read_parquet('data/raw/location.parquet')
print(f"Loaded location data with {len(df)} stations")
print(f"Columns in dataset: {df.columns.tolist()}")

# Check the first few stations
print("\nFirst few stations:")
print(df.head())

# Create GeoDataFrame for stations directly from the DataFrame
# Assuming there are 'longitude' and 'latitude' columns in the data
station_points = []
station_names = []

# Determine the station ID column (might be named 'id', 'station_id', etc.)
id_column = None
for possible_id in ['id', 'station_id', 'station', 'name', 'station_name']:
    if possible_id in df.columns:
        id_column = possible_id
        break

# If no obvious ID column is found, use the index
if id_column is None:
    print("No obvious station ID column found, using index as station ID")
    df['temp_id'] = df.index.astype(str)
    id_column = 'temp_id'

# Create points from the longitude and latitude columns
try:
    # Check for various possible column names
    lon_col = next((col for col in df.columns if col.lower() in ['longitude', 'lon', 'long']), None)
    lat_col = next((col for col in df.columns if col.lower() in ['latitude', 'lat']), None)
    
    if lon_col is None or lat_col is None:
        raise ValueError(f"Could not identify longitude and latitude columns. Available columns: {df.columns.tolist()}")
    
    print(f"Using column '{lon_col}' for longitude and '{lat_col}' for latitude")
    print(f"Using column '{id_column}' for station IDs")
    
    # Create GeoDataFrame from the location data
    stations_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"  # WGS84 coordinate system
    )
    
    print(f"Created GeoDataFrame with {len(stations_gdf)} stations")
except Exception as e:
    print(f"Error creating station points: {e}")
    # Fallback to creating example points
    print("Falling back to example station coordinates")
    for i, idx in enumerate(df.index):
        # Create example coordinates spread across Denmark
        lon = 9.0 + (i * 0.5) % 4  # Spread across longitude
        lat = 55.0 + (i * 0.3) % 2  # Spread across latitude
        station_points.append(Point(lon, lat))
        station_names.append(str(df.loc[idx, id_column]) if id_column in df.columns else f"Station_{idx}")
    
    stations_gdf = gpd.GeoDataFrame(
        {id_column: station_names},
        geometry=station_points,
        crs="EPSG:4326"  # WGS84 coordinate system
    )

# Create simple Denmark boundary if not available
from shapely.geometry import Polygon
denmark_coords = [
    (8.0, 54.5),   # Southwest corner
    (8.0, 57.8),   # Northwest corner
    (13.0, 57.8),  # Northeast corner
    (13.0, 54.5),  # Southeast corner
    (8.0, 54.5)    # Close the polygon
]
denmark_polygon = Polygon(denmark_coords)
denmark = gpd.GeoDataFrame({'name': ['Denmark']}, geometry=[denmark_polygon], crs="EPSG:4326")

# Function to create Voronoi polygons
def create_voronoi_map(stations_gdf, denmark_gdf, output_path='voronoi_map.png', show_plot=True):
    """Create a map showing Voronoi polygons for weather stations in Denmark."""
    import numpy as np
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon
    import contextily as ctx
    
    # Reproject to a projected CRS
    denmark_projected = denmark_gdf.to_crs("EPSG:3857")  # Web Mercator
    stations_projected = stations_gdf.to_crs("EPSG:3857")
    
    # Extract coordinates
    coords = np.array([(p.x, p.y) for p in stations_projected.geometry])
    
    # Get Denmark boundary
    # Fix for deprecation warning: Use union_all() instead of unary_union
    boundary = denmark_projected.geometry.union_all().bounds
    boundary_width = boundary[2] - boundary[0]
    boundary_height = boundary[3] - boundary[1]
    
    # Add corner points far outside actual area to ensure complete coverage
    corner_points = [
        [boundary[0] - boundary_width, boundary[1] - boundary_height],
        [boundary[2] + boundary_width, boundary[1] - boundary_height],
        [boundary[0] - boundary_width, boundary[3] + boundary_height],
        [boundary[2] + boundary_width, boundary[3] + boundary_height]
    ]
    
    all_points = np.vstack([coords, corner_points])
    vor = Voronoi(all_points)
    
    # Function to create finite Voronoi polygons
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
    
    # Get Voronoi polygons
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # Create clipped polygons for each station
    voronoi_polygons = []
    valid_station_ids = []
    
    for i, region in enumerate(regions):
        if i < len(coords):  # Skip corner points
            polygon = Polygon([vertices[v] for v in region])
            # Clip polygon to Denmark boundary - use union_all() instead of unary_union
            clipped_polygon = polygon.intersection(denmark_projected.geometry.union_all())
            if not clipped_polygon.is_empty:
                voronoi_polygons.append(clipped_polygon)
                valid_station_ids.append(stations_projected['station_id'].iloc[i])
    
    # Create GeoDataFrame with coverage areas
    coverage_gdf = gpd.GeoDataFrame(
        {'station_id': valid_station_ids},
        geometry=voronoi_polygons,
        crs=stations_projected.crs
    )
    
    # Plot the map
    fig, ax = plt.subplots(figsize=(12, 10))
    denmark_projected.boundary.plot(ax=ax, color='black', linewidth=1)
    coverage_gdf.plot(ax=ax, alpha=0.5, edgecolor='black', column='station_id', cmap='tab10')
    stations_projected.plot(ax=ax, color='red', markersize=50)
    
    # Add station labels
    for idx, row in stations_projected.iterrows():
        plt.annotate(row['station_id'], xy=(row.geometry.x, row.geometry.y), 
                     xytext=(3, 3), textcoords="offset points")
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=stations_projected.crs, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Could not add basemap: {e}")
    
    plt.title('Weather Station Coverage Areas in Denmark')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Voronoi map to {output_path}")
    
    if show_plot:
        plt.show()
    
    return coverage_gdf

# Execute the function to create and display the Voronoi map
if __name__ == "__main__":
    coverage_areas = create_voronoi_map(stations_gdf, denmark, 'station_coverage_map.png', True)
    print(f"Created coverage areas for {len(coverage_areas)} stations")
