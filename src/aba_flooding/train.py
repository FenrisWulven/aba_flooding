import aba_flooding.model as md
import pandas as pd
# import geopandas as gpd
# import geo_utils as gu
import matplotlib.pyplot as plt
import os
import time
import cProfile
import pstats
from io import StringIO
from aba_flooding.preprocess import load_saved_data
from lifelines import KaplanMeierFitter
import multiprocessing
from functools import partial

###################################
## PROCESS A SINGLE STATION FILE ##
###################################
def process_station_file(file, processed_data_path, profile=False):
    """
    Process a single station file - can be run in parallel
    
    Returns:
    --------
    tuple: (station ID, survival models dict, timing info dict)
    """
    try:
        # Load the survival data from all available station files

        station_start_time = time.time()
        
        # Extract station ID from filename
        station = file.replace("survival_data_", "").replace(".parquet", "")
        
        # Load the data for the station
        file_path = os.path.join(processed_data_path, file)
        
        load_start_time = time.time()
        survival_df = load_saved_data(file_path)
        load_time = time.time() - load_start_time
        
        if survival_df is None or survival_df.empty:
            print(f"Skipping empty data file for station {station}")
            return station, None, {
                'station': station,
                'status': 'skipped',
                'reason': 'empty data'
            }
 
        # Identify soil types within the station
        soil_types = set()
        for column in survival_df.columns:
            parts = column.split('_')
            # Check if this is a station-soil column
            if len(parts) >= 3 and parts[0] == station:
                if parts[1] != "WOG":  # Skip WOG columns
                    soil_types.add(parts[1])
        
        # Create a dictionary for all survival models within the station
        station_models = {}
        
        # Run profiling if enabled
        if profile:
            print(f"\nProfiling station {station}...")
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Process each soil type
            for soil_type in soil_types:
                # Create column names
                duration_column = f"{station}_{soil_type}_duration"
                event_column = f"{station}_{soil_type}_observed"
                
                # Check if columns exist
                if duration_column in survival_df.columns and event_column in survival_df.columns:
                    valid_data = survival_df[[duration_column, event_column]].dropna()
                    
                    if len(valid_data) > 0:
                        # Create and train the model
                        model = md.SurvivalModel(soil_type=soil_type)
                        model.station = station
                        model.train(
                            valid_data.rename(columns={
                                duration_column: 'duration',
                                event_column: 'observed'
                            }),
                            'duration', 
                            'observed'
                        )
                        
                        # Add to our local models to dictionary
                        model_key = f"{station}_{soil_type}"
                        station_models[model_key] = model
            
            profiler.disable()
            
            # Print profile results
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions by time
            print(s.getvalue())
        else:
            # Add station to the model without profiling
            train_start_time = time.time()
            
            # Process each soil type within the station boundry
            for soil_type in soil_types:
                # Create column names
                duration_column = f"{station}_{soil_type}_duration"
                event_column = f"{station}_{soil_type}_observed"
                
                # Check if columns exist (ensure that everything worked so far)
                if duration_column in survival_df.columns and event_column in survival_df.columns:
                    valid_data = survival_df[[duration_column, event_column]].dropna()
                    
                    if len(valid_data) > 0:
                        # Create and train the model
                        model = md.SurvivalModel(soil_type=soil_type)
                        model.station = station
                        model.train(
                            valid_data.rename(columns={
                                duration_column: 'duration',
                                event_column: 'observed'
                            }),
                            'duration', 
                            'observed'
                        )
                        
                        # Add to our local models dictionary
                        model_key = f"{station}_{soil_type}"
                        station_models[model_key] = model
            
            train_time = time.time() - train_start_time
        
        station_time = time.time() - station_start_time
        
        # Prepare timing info for profiling
        timing = {
            'station': station,
            'total_time': station_time,
            'load_time': load_time,
            'train_time': station_time - load_time,
            'soil_types': len(soil_types),
            'models_trained': len(station_models)
        }
        
        print(f"Station {station}: {station_time:.2f}s (Load: {load_time:.2f}s, Train: {station_time - load_time:.2f}s)")
        
        return station, station_models, timing
        
    except Exception as e:
        print(f"Error processing station file {file}: {e}")
        import traceback
        traceback.print_exc()
        return file, None, {'status': 'error', 'error': str(e)}

##########################################
## TRAINING ALL MODELS FOR ALL STATIONS ##
##########################################
def train_all_models(output_path="models/flood_model.pkl", profile=False, parallel=True, max_workers=None):
    """
    Train survival models for all stations and soil types from processed parquet files.
    
    Parameters:
    -----------
    output_path : str
        Path where the trained model will be saved
    profile : bool
        Whether to run detailed profiling for each station
    parallel : bool
        Whether to use parallel processing (default: True)
    max_workers : int or None
        Maximum number of parallel workers (default: CPU count - 1)
        
    Returns:
    --------
    FloodModel : Trained flood model
    dict : Timing information
    """
    start_time = time.time()
    timing_info = {'total': 0, 'stations': {}}
    
    # Find all files in data/processed/
    # This is the designated destination for the survival data and will not change
    processed_data_path = os.path.join(os.getcwd(), "data/processed/")
    station_files = []

    # Check that data/processed/ exists if not then throw an error
    if os.path.exists(processed_data_path):
        # List all files in the directory (all available stations)
        files = os.listdir(processed_data_path)
        
        # Filter for parquet files (making sure to only get preprocessed data)
        station_files = [f for f in files if f.startswith("survival_data_") and f.endswith(".parquet")]
        print(f"Found {len(station_files)} station data files")
    else:
        print(f"Directory {processed_data_path} does not exist")
        return None, timing_info

    # Create a new FloodModel
    flood_model = md.FloodModel()

    # Use parallel processing if enabled
    if parallel and len(station_files) > 1 and not profile:  # Skip parallel if profiling
        # Determine number of workers - default is based on physical cores when possible
        if max_workers is None:
            try:
                import psutil
                # Get physical cores not logical cores as we need to read from disk
                physical_cores = psutil.cpu_count(logical=False)
                max_workers = max(1, physical_cores - 2) # Leave room for system
            except (ImportError, AttributeError):
                # If psutil isn't available, use a conservative default (about 60% of logical processors)
                max_workers = max(1, int(multiprocessing.cpu_count() * 0.6))
                
            # Cap at 8 workers to prevent overloading
            max_workers = min(8, max_workers)
        
        print(f"Using parallel processing with {max_workers} workers")
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=max_workers) as pool:
            # Process all stations in parallel
            process_func = partial(process_station_file, 
                                  processed_data_path=processed_data_path, 
                                  profile=profile)
            
            results = pool.map(process_func, station_files)
            
            # Collect results
            for station, models, timing in results:
                if models:
                    # Add all models to the main flood model
                    for model_key, model in models.items():
                        flood_model.models[model_key] = model
                    
                    # Update station list if not already there
                    if station not in flood_model.stations:
                        flood_model.stations.append(station)
                    
                    # Update soil types
                    for model_key in models:
                        soil_type = model_key.split('_')[1]
                        if soil_type not in flood_model.available_soil_types:
                            flood_model.available_soil_types.append(soil_type)
                
                # Store timing info
                if isinstance(timing, dict) and 'station' in timing:
                    timing_info['stations'][station] = timing
    else:
        # Process files sequentially
        for file in station_files:
            station, models, timing = process_station_file(file, processed_data_path, profile)
            
            if models:
                # Add all models to the main flood model
                for model_key, model in models.items():
                    flood_model.models[model_key] = model
                
                # Update station list if not already there
                if station not in flood_model.stations:
                    flood_model.stations.append(station)
                
                # Update soil types
                for model_key in models:
                    soil_type = model_key.split('_')[1]
                    if soil_type not in flood_model.available_soil_types:
                        flood_model.available_soil_types.append(soil_type)
            
            # Store timing info
            if isinstance(timing, dict) and 'station' in timing:
                timing_info['stations'][station] = timing

    # If we've trained models then 
    if flood_model.models:
        flood_model.is_fitted = True
        total_time = time.time() - start_time
        timing_info['total'] = total_time
    else:
        print("No models were successfully trained")
        total_time = time.time() - start_time
        timing_info['total'] = total_time

    return flood_model, timing_info

def print_timing_report(timing_info):
    """Print a formatted timing report from timing information."""
    if not timing_info:
        print("No timing information available")
        return
        
    print("\n" + "="*60)
    print("TRAINING PERFORMANCE REPORT")
    print("="*60)
    print(f"Total training time: {timing_info['total']:.2f} seconds")
    print(f"Total stations: {len(timing_info['stations'])}")
    
    if timing_info['stations']:
        # Overall stats
        station_times = [info['total_time'] for info in timing_info['stations'].values()]
        avg_time = sum(station_times) / len(station_times)
        max_time = max(station_times)
        min_time = min(station_times)
        
        print(f"\nAverage time per station: {avg_time:.2f}s")
        print(f"Fastest station: {min_time:.2f}s")
        print(f"Slowest station: {max_time:.2f}s")
        
        # Top 5 slowest stations
        print("\nTop 5 slowest stations:")
        sorted_stations = sorted(timing_info['stations'].items(), 
                                key=lambda x: x[1]['total_time'], 
                                reverse=True)
        
        for i, (station, info) in enumerate(sorted_stations[:5], 1):
            print(f"{i}. Station {station}: {info['total_time']:.2f}s - {info['soil_types']} soil types")
            
        # Print save time if available
        if 'save_time' in timing_info:
            print(f"\nModel save time: {timing_info['save_time']:.2f}s")
    
    print("="*60)


if __name__ == "__main__":
    # Make sure the output directory exists
    os.makedirs("models", exist_ok=True)
    
    # Enable detailed profiling if needed
    enable_profiling = False
    
    # Enable parallel processing
    enable_parallel = False
    
    # Set number of workers (None = auto)
    num_workers = None
    
    # Train models for all stations
    print("Training models for all stations...")
    training_start = time.time()
    flood_model, timing_info = train_all_models(
        "models/flood_model.joblib",
        profile=enable_profiling,
        parallel=enable_parallel, 
        max_workers=None  # Default to CPU count - 2
    )

    # Save using split storage
    print("Saving model with split storage...")
    save_start = time.time()
    flood_model.save("models/flood_model.joblib", split_by_station=True)
    save_time = time.time() - save_start
    print(f"Model saved in {save_time:.2f} seconds")
    # add save time to timing info
    timing_info['save_time'] = save_time

    # Print the timing report
    print_timing_report(timing_info)