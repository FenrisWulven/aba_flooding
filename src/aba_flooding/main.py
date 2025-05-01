import aba_flooding.preprocess as p
import aba_flooding.train as t
import aba_flooding.map as map
import aba_flooding.inspect_model as im
import pandas as pd
import aba_flooding.geo_utils as gu


precipitation_data = pd.read_parquet("data/raw/precipitation_imputed_data.parquet")
for column in precipitation_data.columns:
    if precipitation_data[column].count() < 5000:
        print(f"Removing column {column} with {precipitation_data[column].count()} non-NaN values")
        precipitation_data.drop(column, axis=1, inplace=True)
# clip
precipitation_data = precipitation_data.clip(lower=0, upper=100) # this is already done in the create_precipitation_coverage function
coverage_geojson_gdf, stations_gdf = p.create_full_coverage(precipitation_data)
if coverage_geojson_gdf is None:
    print("ERROR No valid coverage data created. Attemption to load from file.")
    try:
        coverage_geojson_gdf = gu.load_geojson("precipitation_coverage.geojson")
        print("Loaded coverage data from file.")
    except Exception as e:
        print(f"ERROR loading coverage data from file: {e}")
        print("Exiting.")
        exit(1)

print("Loading Sediment_wgs84.geojson...")
sedimentCoverage = gu.load_geojson("Sediment_wgs84.geojson")
if sedimentCoverage is None:
    print("No valid sediment data loaded. Exiting.")
    exit(1)

p.load_process_data(coverage_data=coverage_geojson_gdf, sediment_data=sedimentCoverage)

print("Perculation mapping done.")

flood_model, timing_info = t.train_all_models(
        "models/flood_model.joblib",
        profile=False,
        parallel=True, 
        max_workers=None  # Default to CPU count - 2
    )

print("Flood model training completed.")

im.inspect_model(False)

print("Model inspection completed.")
print("Creating map...")

from bokeh.io import output_file, show

p = map.init_map()
# Save to an HTML file and display in browser
output_file("terrain_map.html")
show(p)

p = map.init_map(sjælland=True)
# Save to an HTML file and display in browser
output_file("terrain_map_sjaelland.html")
show(p)