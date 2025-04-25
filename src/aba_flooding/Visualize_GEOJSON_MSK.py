import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

def visualize_denmark_sediments(geojson_file_path):
    # Load the GeoJSON file
    print("Loading GeoJSON file...")
    gdf = gpd.read_file(geojson_file_path)
    
    # Print basic information about the dataset
    print(f"Data loaded. Number of features: {len(gdf)}")
    print("Columns in the dataset:", gdf.columns.tolist())
    
    # Extract unique sediment types
    sediment_types = gdf['sediment'].unique()
    print(f"Found {len(sediment_types)} unique sediment types")
    
    # Create a color map for the sediment types
    # Using a colorblind-friendly palette
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    sediment_colors = {sed_type: colors[i % len(colors)] for i, sed_type in enumerate(sediment_types)}
    
    # Create a figure and axis
    fig, ax = plt.figure(figsize=(15, 15)), plt.gca()
    
    # Plot the map
    print("Plotting the map (this may take some time for large files)...")
    for sediment_type in sediment_types:
        subset = gdf[gdf['sediment'] == sediment_type]
        subset.plot(
            ax=ax,
            color=sediment_colors[sediment_type],
            edgecolor='black',
            linewidth=0.3,
            alpha=0.7
        )
    
    # Create a legend
    # For large numbers of sediment types, create a separate legend figure
    if len(sediment_types) > 15:
        print("Creating separate legend due to large number of sediment types...")
        create_separate_legend(sediment_types, sediment_colors)
    else:
        # Create patches for the legend
        patches = [mpatches.Patch(color=sediment_colors[sed], label=sed) for sed in sediment_types]
        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')
    
    # Set title and layout
    plt.title('Denmark Sediment Types', fontsize=16)
    plt.axis('off')  # Remove axes
    plt.tight_layout()
    
    # Save the figure
    output_file = "denmark_sediment_map.png"
    print(f"Saving map to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_file}")
    
    # Show the plot
    plt.show()

def create_separate_legend(sediment_types, sediment_colors):
    """Create a separate figure for the legend when there are many sediment types"""
    # Calculate the number of rows and columns for the legend
    n_items = len(sediment_types)
    n_cols = 2
    n_rows = np.ceil(n_items / n_cols).astype(int)
    
    # Create a new figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(10, n_rows * 0.5))
    
    # Create patches for each sediment type
    patches = [mpatches.Patch(color=sediment_colors[sed], label=sed) for sed in sediment_types]
    
    # Add the legend to the figure
    ax_legend.legend(handles=patches, loc='center', ncol=n_cols)
    ax_legend.axis('off')  # Turn off axis
    
    # Save the legend
    fig_legend.savefig("denmark_sediment_legend.png", dpi=300, bbox_inches='tight')
    print("Separate legend saved to denmark_sediment_legend.png")

def optimize_large_geojson(geojson_file_path, output_file_path=None, simplify_tolerance=0.0001):
    """Simplify the GeoJSON file to make it smaller and faster to process"""
    if output_file_path is None:
        output_file_path = geojson_file_path.replace('.geojson', '_simplified.geojson')
    
    print("Loading GeoJSON for simplification...")
    gdf = gpd.read_file(geojson_file_path)
    
    print(f"Original feature count: {len(gdf)}")
    print(f"Simplifying geometries with tolerance {simplify_tolerance}...")
    
    # Simplify the geometries
    gdf['geometry'] = gdf['geometry'].simplify(simplify_tolerance, preserve_topology=True)
    
    # Save the simplified GeoJSON
    print(f"Saving simplified GeoJSON to {output_file_path}...")
    gdf.to_file(output_file_path, driver='GeoJSON')
    
    print("Simplification complete!")
    return output_file_path

if __name__ == "__main__":
    # Path to your GeoJSON file
    geojson_file = "/Users/maks/Documents/GitHub/aba_flooding/data/raw/Sediment_wgs84.geojson"
    
    # If the file is very large, you might want to simplify it first
    # Uncomment the following lines to use this feature
    # simplified_file = optimize_large_geojson(geojson_file, simplify_tolerance=0.0001)
    # visualize_denmark_sediments(simplified_file)
    
    # Or directly visualize the original file
    visualize_denmark_sediments(geojson_file)