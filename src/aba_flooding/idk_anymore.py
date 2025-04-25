
#=================================================================================================#
# This script retrieves weather data from the DMI API for all stations in Denmark.
# It saves the data to JSON files and creates a map visualization using Folium.
# It handles pagination, retries, and error handling for API requests.
# It also includes logging for better tracking of the process.
#=================================================================================================#

import requests, os, json, folium, logging, time, random, gc, time, logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dmi_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Define your API key
api_key = 'd111ba1d-a1f5-43a5-98c6-347e9c2729b2'  # Replace with your actual DMI API key

# Memory and performance settings
MAX_MEMORY_USAGE_GB = 14     # Increased for maximum performance
MAX_THREADS = 8              # Increased for better parallelism
RATE_LIMIT_DELAY = 0.5       # Reduced for faster data collection
MAX_RETRIES = 100             # High number of retries for resilience
RETRY_DELAY = 20             # Reduced initial retry delay
EXPONENTIAL_BACKOFF = True   # Still using exponential backoff to handle rate limits

# Define the output directory
output_dir = './dmi_data_daily'  # Changed to relative path for portability
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the bounding box for Denmark [min_lon, min_lat, max_lon, max_lat]
denmark_bbox = [7.253075, 54.303704, 13.321548, 57.809651]

# Define the parameters to retrieve
parameters = [
    "precip_past1h"  # Precipitation in the last hour
]

# Function to save current state of all station data - MOVED UP before it's used
def save_station_data(stations_dict, output_directory, current_parameter):
    """Save all station data to individual JSON files."""
    logger.info("Saving current data to files...")
    saved_count = 0
    
    # Create a parameters directory to store parameter-specific files
    params_dir = os.path.join(output_directory, 'parameters')
    os.makedirs(params_dir, exist_ok=True)
    
    # First, save a parameter-specific file (for recovery if needed)
    param_file = os.path.join(params_dir, f'parameter_{current_parameter}.json')
    stations_with_param = {}
    for station_id, station_data in stations_dict.items():
        if current_parameter in station_data.get('parameters', {}):
            # Create a copy with only this parameter's data
            param_station = {
                'stationId': station_data.get('stationId'),
                'name': station_data.get('name', ''),
                'location': station_data.get('location', {}),
                'parameters': {current_parameter: station_data['parameters'][current_parameter]}
            }
            stations_with_param[station_id] = param_station
    
    # Save the parameter-specific file
    with open(param_file, 'w', encoding='utf-8') as f:
        json.dump(stations_with_param, f)
    
    # Now save individual station files with all accumulated data
    stations_dir = os.path.join(output_directory, 'stations')
    os.makedirs(stations_dir, exist_ok=True)
    
    for station_id, station_data in tqdm(stations_dict.items(), desc="Saving station data"):
        if station_data.get('parameters'):
            # Save to JSON file
            filename = os.path.join(stations_dir, f'station_{station_id}.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(station_data, f, indent=2)
            saved_count += 1
    
    logger.info(f"Saved data for {saved_count} stations to {output_directory}")
    
    # Also save a parameter progress file
    progress_file = os.path.join(output_directory, 'parameter_progress.json')
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump({
            'last_processed_parameter': current_parameter, 
            'timestamp': datetime.now().isoformat(),
            'stations_saved': saved_count
        }, f, indent=2)
    
    return saved_count

# Calculate the time frame
end_time = pd.Timestamp.now(tz='UTC')
start_time = end_time - pd.DateOffset(years=30)  # Start 30 years ago
datetime_str = f"{start_time.isoformat()}/{end_time.isoformat()}"

# Function to retrieve all stations with retry logic
def get_all_stations(api_key):
    """Retrieve all DMI stations, handling pagination and retries."""
    url = 'https://dmigw.govcloud.dk/v2/metObs/collections/station/items'
    params = {'api-key': api_key, 'limit': '10000'}
    stations = []
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            logger.info(f"Retrieving stations (attempt {retry_count + 1}/{MAX_RETRIES})...")
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            json_data = r.json()
            stations.extend(json_data['features'])
            
            next_link = next((link for link in json_data['links'] if link['rel'] == 'next'), None)
            if next_link:
                logger.info(f"Found next page link, continuing pagination...")
                url = next_link['href']
                params = {}  # Clear params for subsequent requests
            else:
                logger.info(f"Station retrieval complete. Found {len(stations)} stations.")
                break
                
            # Add delay to avoid rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except requests.RequestException as e:
            logger.error(f"Error retrieving stations: {e}")
            retry_count += 1
            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Maximum retry attempts reached. Proceeding with collected stations.")
                break
    
    return stations

# Function to get data for a specific parameter with improved error handling
def get_data_for_parameter(parameter_id, datetime_str, api_key, bbox=None, time_chunks=1):
    """Retrieve data for a specific parameter, with robust error handling and retry logic."""
    all_data = []
    
    # Parse start and end times
    times = datetime_str.split('/')
    start_time = pd.Timestamp(times[0])
    end_time = pd.Timestamp(times[1])
    
    # Calculate the time delta for each chunk
    total_days = (end_time - start_time).days
    days_per_chunk = max(1, total_days // time_chunks)
    
    logger.info(f"Splitting timeframe into {time_chunks} chunks of approximately {days_per_chunk} days each")
    
    # Process each time chunk
    for i in range(time_chunks):
        chunk_start = start_time + pd.Timedelta(days=i * days_per_chunk)
        chunk_end = start_time + pd.Timedelta(days=(i+1) * days_per_chunk) if i < time_chunks - 1 else end_time
        chunk_datetime_str = f"{chunk_start.isoformat()}/{chunk_end.isoformat()}"
        
        logger.info(f"Processing time chunk {i+1}/{time_chunks}: {chunk_start.date()} to {chunk_end.date()}")
        
        # Set up the request parameters
        url = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items'
        params = {
            'api-key': api_key,
            'datetime': chunk_datetime_str,
            'parameterId': parameter_id,
            'limit': '10000'  # Reduced limit to minimize server errors
        }
        
        # Add bbox parameter if provided
        if bbox:
            params['bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        # Variables to track pagination
        offset = 0
        max_offset = 490000  # Stay below the 500,000 limit
        has_more = True
        chunk_data = []
        
        # Retry loop
        while has_more and offset < max_offset:
            retry_count = 0
            success = False
            
            while retry_count < MAX_RETRIES and not success:
                try:
                    # Add the offset parameter for pagination
                    if offset > 0:
                        params['offset'] = str(offset)
                    
                    logger.info(f"Making request to: {url} with offset {offset} (attempt {retry_count + 1}/{MAX_RETRIES})")
                    r = requests.get(url, params=params, timeout=120)  # Increased timeout
                    r.raise_for_status()
                    json_data = r.json()
                    
                    if 'features' in json_data:
                        batch_size = len(json_data['features'])
                        logger.info(f"Retrieved {batch_size} records in this batch")
                        chunk_data.extend(json_data['features'])
                        
                        # Save data periodically to avoid memory issues
                        if len(chunk_data) >= 100000:  # Increased batch size before saving
                            logger.info("Saving intermediate batch to avoid memory issues...")
                            save_parameter_batch(parameter_id, chunk_data, chunk_datetime_str)
                            all_data.extend(chunk_data)  # Add to total count
                            chunk_data = []  # Clear for next batch
                            gc.collect()  # Force garbage collection
                        
                        # Check if we need to continue pagination
                        if batch_size < int(params['limit']):
                            has_more = False
                        else:
                            offset += batch_size
                        
                        success = True
                    else:
                        logger.warning(f"No 'features' in response. Response: {json_data}")
                        has_more = False
                        success = True
                    
                except requests.RequestException as e:
                    error_msg = str(e)
                    logger.error(f"Error retrieving data: {error_msg}")
                    
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        response_text = e.response.text if hasattr(e.response, 'text') else "No response text"
                        logger.error(f"Status code: {status_code}, Response: {response_text}")
                        
                        # Handle specific errors
                        if status_code == 429:  # Too Many Requests
                            logger.warning("Rate limit exceeded. Increasing wait time.")
                            time.sleep(RETRY_DELAY * 2)  # Double the retry delay
                        elif status_code in [502, 503, 504]:  # Server errors
                            logger.warning(f"Server error {status_code}. Will retry.")
                        elif status_code == 400 and 'Offset cannot be greater than 500000' in response_text:
                            logger.warning("Hit offset limit. Saving current batch and continuing with next time chunk.")
                            has_more = False
                            success = True  # Exit retry loop but not while loop
                    
                    retry_count += 1
                    if retry_count < MAX_RETRIES and not success:
                        if EXPONENTIAL_BACKOFF:
                            # Exponential backoff with jitter
                            wait_time = min(RETRY_DELAY * (2 ** (retry_count - 1)) + (random.randint(0, 1000) / 1000), 600)
                        else:
                            wait_time = RETRY_DELAY * retry_count  # Linear backoff
                        
                        logger.info(f"Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error("Maximum retry attempts reached for this batch.")
                        has_more = False  # Stop trying this chunk
                
                # Add a delay between requests to avoid rate limiting
                time.sleep(RATE_LIMIT_DELAY)
        
        # Save any remaining data from this chunk
        if chunk_data:
            logger.info(f"Saving final batch of {len(chunk_data)} records for time chunk {i+1}")
            save_parameter_batch(parameter_id, chunk_data, chunk_datetime_str)
            all_data.extend(chunk_data)
            chunk_data = []
            gc.collect()
    
    total_records = len(all_data)
    logger.info(f"Total records retrieved for {parameter_id}: {total_records}")
    return all_data

# Function to save a batch of parameter data
def save_parameter_batch(parameter_id, batch_data, datetime_str):
    """Save a batch of parameter data to a file."""
    if not batch_data:
        return
    
    batch_size = len(batch_data)
    logger.info(f"Saving batch of {batch_size} records for {parameter_id}...")
    
    # Create a batch directory
    batch_dir = os.path.join(output_dir, 'parameter_batches', parameter_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    # Generate a unique batch filename based on timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_file = os.path.join(batch_dir, f'{parameter_id}_batch_{timestamp}.json')
    
    # Save the batch with metadata
    batch_metadata = {
        'parameter_id': parameter_id,
        'datetime_range': datetime_str,
        'record_count': batch_size,
        'created_at': datetime.now().isoformat(),
        'data': batch_data
    }
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_metadata, f)
    
    logger.info(f"Saved batch to {batch_file}")
    
    # Update the parameter tracking file
    update_parameter_tracking(parameter_id, batch_size, batch_file, datetime_str)

# Function to update the parameter tracking file
def update_parameter_tracking(parameter_id, batch_size, batch_file, datetime_str):
    """Update the parameter tracking file with new batch information."""
    tracking_file = os.path.join(output_dir, 'parameter_tracking.json')
    tracking_data = {}
    
    # Load existing tracking data if it exists
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                tracking_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading tracking file: {e}")
    
    # Update the tracking data
    if parameter_id not in tracking_data:
        tracking_data[parameter_id] = {
            'total_records': 0,
            'batches': []
        }
    
    tracking_data[parameter_id]['total_records'] += batch_size
    tracking_data[parameter_id]['batches'].append({
        'file': os.path.basename(batch_file),
        'records': batch_size,
        'datetime_range': datetime_str,
        'created_at': datetime.now().isoformat()
    })
    
    # Save updated tracking data
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2)

# Function to combine parameter batches
def combine_parameter_batches(parameter_id):
    """Combine all stored batches for a parameter into station data."""
    batch_dir = os.path.join(output_dir, 'parameter_batches', parameter_id)
    if not os.path.exists(batch_dir):
        logger.warning(f"No batch directory found for parameter {parameter_id}")
        return {}
    
    # Dictionary to hold station data
    stations_data = {}
    
    # List all batch files
    batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.json')]
    if not batch_files:
        logger.warning(f"No batch files found for parameter {parameter_id}")
        return {}
    
    logger.info(f"Found {len(batch_files)} batch files for parameter {parameter_id}")
    
    # Process each batch file
    for batch_file in tqdm(batch_files, desc=f"Processing {parameter_id} batches"):
        try:
            with open(os.path.join(batch_dir, batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            # Process each record in the batch
            for record in batch_data['data']:
                station_id = record['properties']['stationId']
                
                # Initialize station if not exists
                if station_id not in stations_data:
                    stations_data[station_id] = {
                        'stationId': station_id,
                        'parameters': {parameter_id: []}
                    }
                
                # Initialize parameter if not exists
                if parameter_id not in stations_data[station_id]['parameters']:
                    stations_data[station_id]['parameters'][parameter_id] = []
                
                # Add the record
                stations_data[station_id]['parameters'][parameter_id].append(record)
        except Exception as e:
            logger.error(f"Error processing batch file {batch_file}: {e}")
    
    return stations_data

# Main execution function
def main():
    # Retrieve all stations
    logger.info("Starting DMI data collection process")
    all_stations = get_all_stations(api_key)

    # Filter stations within the bounding box
    filtered_stations = {}
    for station in all_stations:
        coords = station['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        if (denmark_bbox[0] <= lon <= denmark_bbox[2] and 
            denmark_bbox[1] <= lat <= denmark_bbox[3]):
            station_id = station['properties']['stationId']
            filtered_stations[station_id] = {
                'stationId': station_id,
                'name': station['properties'].get('name', ''),
                'location': {
                    'longitude': lon,
                    'latitude': lat
                },
                'parameters': {}
            }

    logger.info(f"Found {len(filtered_stations)} stations within the bounding box.")

    # Check for existing progress file
    progress_file = os.path.join(output_dir, 'parameter_progress.json')
    last_processed_idx = -1  # Start from the beginning by default
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                last_param = progress_data.get('last_processed_parameter')
                if last_param in parameters:
                    last_processed_idx = parameters.index(last_param)
                    logger.info(f"Resuming from parameter: {last_param} (index {last_processed_idx})")
        except Exception as e:
            logger.error(f"Error reading progress file: {e}")
    
    # Create directory for parameter batches
    batch_dir = os.path.join(output_dir, 'parameter_batches')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Process each parameter for all stations at once, starting after the last processed parameter
    for i, parameter in enumerate(parameters[last_processed_idx+1:], start=last_processed_idx+1):
        logger.info(f"\nRetrieving {parameter} data for all stations... ({i+1}/{len(parameters)})")
        
        # Use smaller time chunks for better handling
        time_chunks = 12  # Split into smaller chunks to reduce server load and handle errors better
        
        # First retrieve parameter data in batches
        try:
            logger.info(f"Retrieving data for parameter {parameter} in batches...")
            get_data_for_parameter(parameter, datetime_str, api_key, denmark_bbox, time_chunks=time_chunks)
            
            # Now combine all batches for this parameter
            logger.info(f"Combining batches for parameter {parameter}...")
            parameter_stations = combine_parameter_batches(parameter)
            
            # Merge parameter data with existing station data
            for station_id, station_data in parameter_stations.items():
                if station_id in filtered_stations:
                    # Add station metadata if it's a new station
                    if 'name' not in station_data and 'name' in filtered_stations[station_id]:
                        station_data['name'] = filtered_stations[station_id]['name']
                    if 'location' not in station_data and 'location' in filtered_stations[station_id]:
                        station_data['location'] = filtered_stations[station_id]['location']
                    
                    # Add parameter data to the station
                    if parameter in station_data['parameters']:
                        filtered_stations[station_id]['parameters'][parameter] = station_data['parameters'][parameter]
            
            # Save all station data after each parameter is processed
            logger.info(f"Completed processing parameter: {parameter}")
            save_station_data(filtered_stations, output_dir, parameter)
            
            # Free up memory
            parameter_stations = None
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing parameter {parameter}: {e}")
            # Still save progress even if we encounter an error
            save_station_data(filtered_stations, output_dir, f"{parameter}_error")

    # Final save to ensure everything is written
    logger.info("\nPerforming final data save...")
    save_count = save_station_data(filtered_stations, output_dir, "final")
    
    logger.info(f"Data extraction complete. Saved data for {save_count} stations.")

if __name__ == "__main__":
    try:
        main()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        logger.info("Program terminated with errors")