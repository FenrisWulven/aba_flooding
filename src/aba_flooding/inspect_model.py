import os
import matplotlib.pyplot as plt
import numpy as np

from aba_flooding.train import process_station_file
from aba_flooding.model import FloodModel


def inspect_model(train = False):
    """Inspect the trained flood model."""
    model_path = os.path.join("models", "flood_model.joblib")
    
    
    print(f"Loading model from {model_path}...")
    try:
        # Create a FloodModel instance first, then call load with the path parameter
        model = FloodModel()
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        model.load(path=model_path)
        
        # Inspect specific station
        inspect_station(model, "05135")
        
        print("\n== Model Summary ==")
        print(f"Number of stations: {len(model.stations)}")
        print(f"Total models: {len(model.models)}")
        print(f"Number of available soil types: {len(model.available_soil_types)}")
        print(f"Available soil types: {model.available_soil_types}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

def inspect_station(model, station_id):
    """
    Inspect a specific station's models in detail
    
    Parameters:
    -----------    
    model : FloodModel
        The loaded FloodModel instance
    station_id : str
        The station ID to inspect
    """
    print(f"\n==== INSPECTING STATION {station_id} ====")
    
    # Check if station exists in the model
    if station_id not in model.stations:
        print(f"Station {station_id} not found in model")
        return
    
    # For debugging, let's examine the model structure
    print(f"Available model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")

    # Try using the get_station_models method if available
    if hasattr(model, 'get_station_models'):
        print(f"Using get_station_models to load models for station {station_id}...")
        try:
            station_models = model.get_station_models(station_id)
            if station_models:
                print(f"Successfully loaded {len(station_models)} models for station {station_id}")
                
                # Plot survival curves for this station
                plot_survival_curves(station_models, station_id)
                
                # Report on each model
                for key, survival_model in station_models.items():
                    # Extract just the soil type part (remove station prefix if present)
                    if key.startswith(f"{station_id}_"):
                        soil_type = key.replace(f"{station_id}_", "")
                    else:
                        soil_type = key
                        
                    print(f"\nSoil type: {soil_type}")
                    
                    # Print model attributes for debugging
                    model_attrs = [attr for attr in dir(survival_model) if not attr.startswith('_')]
                    print(f"  - Model attributes: {model_attrs}")
                    
                    # Check if model is fitted
                    is_fitted = survival_model.is_fitted if hasattr(survival_model, 'is_fitted') else False
                    print(f"  - Is fitted: {is_fitted}")
                    
                    # Check for model attribute that might contain the fitted estimator
                    if hasattr(survival_model, 'model') and survival_model.model is not None:
                        print(f"  - Has model object: Yes")
                        if hasattr(survival_model.model, 'median_survival_time_'):
                            print(f"  - Median survival time: {survival_model.model.median_survival_time_}")
                    else:
                        print(f"  - Has model object: No")
                    
                    # Test different time intervals - both in years and days
                    time_periods = [
                        {'value': 10, 'unit': 'days'},
                        {'value': 150, 'unit': 'days'},
                        {'value': 1, 'unit': 'years'},
                        {'value': 5, 'unit': 'years'},
                        {'value': 10, 'unit': 'years'}
                    ]
                    print(f"  - Prediction results:")
                    for period in time_periods:
                        try:
                            if hasattr(survival_model, 'predict_proba'):
                                # Convert days to years for prediction if needed
                                t_value = period['value']
                                if period['unit'] == 'days':
                                    t_years = t_value / 365.25  # Convert to years
                                else:
                                    t_years = t_value
                                

                                surv_prob = survival_model.predict_proba(t_years)
                                # Flood probability is 1-survival probability
                                flood_prob = (surv_prob)*100 #if isinstance(surv_prob, (int, float)) else None
                                print(f"    - At {period['value']} {period['unit']}: {flood_prob:.2f}% flood probability" 
                                      if flood_prob is not None else f"    - At {period['value']} {period['unit']}: No valid prediction")
                            else:
                                print(f"    - At {period['value']} {period['unit']}: predict_proba method not available")
                        except Exception as e:
                            print(f"    - At {period['value']} {period['unit']}: Error - {str(e)}")
                
                return  # Exit once we've used get_station_models successfully
            else:
                print(f"get_station_models returned empty for station {station_id}")
                
        except Exception as e:
            print(f"Error using get_station_models: {e}")
            import traceback
            traceback.print_exc()
    
    # Check for station model files directly
    print("\nChecking for station model files...")
    model_dir = os.path.join("models", "stations")
    station_file = os.path.join(model_dir, f"{station_id}.joblib")
    
    if os.path.exists(station_file):
        print(f"Found station file at {station_file}")
        # You could add code here to load and inspect this file directly
    else:
        print(f"No station file found at {station_file}")
    
    # List all available station files
    if os.path.exists(model_dir):
        station_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if station_files:
            print(f"\nAvailable station files ({len(station_files)} total):")
            for i, file in enumerate(sorted(station_files)[:10]):
                print(f"  - {file}")
            if len(station_files) > 10:
                print(f"  - ... and {len(station_files) - 10} more")
        else:
            print("No station files found in models/stations directory")
    else:
        print(f"Directory {model_dir} does not exist")
    
    print("\n==== STATION INSPECTION COMPLETE ====")

def plot_survival_curves(station_models, station_id):
    """
    Plot survival curves for models of a specific station
    
    Parameters:
    -----------    
    station_models : dict
        Dictionary of soil type -> survival model
    station_id : str
        The station ID
    """
    print("\n== Creating Survival Curve Plots ==")
    
    if not station_models:
        print("No models available to plot")
        return
    
    # Create output directory if it doesn't exist
    plot_dir = os.path.join("outputs", "plots", "inspect_model")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Different time scales to plot
    time_ranges = [
        {"max": 1, "label": "1 Year", "filename": "1year"}
    ]
    
    # For each time range, create a separate plot
    for time_range in time_ranges:
        plt.figure(figsize=(10, 6))
        
        # Generate time points (convert to days for x-axis display)
        t_max = 200000  # in hours
        t = np.linspace(1, t_max, 200)  # 100 points from 0.01 to max years (avoid 0)
        t_days = t / 24 / 365.25  # Convert to years
        
        soil_types_plotted = 0
        
        # Plot each soil type
        for key, survival_model in station_models.items():
            # Extract soil type for the legend
            if key.startswith(f"{station_id}_"):
                soil_type = key.replace(f"{station_id}_", "")
            else:
                soil_type = key
                
            try:
                if hasattr(survival_model, 'predict_proba') and hasattr(survival_model, 'is_fitted') and survival_model.is_fitted:
                    # Get survival probabilities at each time point
                    survival_probs = [survival_model.predict_proba(time_point) for time_point in t]
                    
                    # Convert to flood probabilities
                    flood_probs = [prob for prob in survival_probs]
                    
                    # Plot the flood probability curve
                    plt.plot(t_days, flood_probs, label=f"Soil: {soil_type}", linewidth=2)
                    soil_types_plotted += 1
            except Exception as e:
                print(f"  Error plotting model for soil type {soil_type}: {e}")
        
        if soil_types_plotted > 0:
            plt.xlabel('Time (years)')
            plt.ylabel('Flood Probability')
            plt.title(f'Flood Probability Curves for Station {station_id}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=8) 
            
            # Save the plot
            filename = f"station_{station_id}_flood_prob_{time_range['filename']}.png"
            filepath = os.path.join(plot_dir, filename)
            plt.savefig(filepath)
            print(f"  Saved plot: {filepath}")
            
            plt.close()
        else:
            plt.close()
            print(f"  No valid models to plot for {time_range['label']} time range")
    
    print("== Plotting complete ==")

import pandas as pd
from lifelines import KaplanMeierFitter, WeibullFitter, ExponentialFitter, LogNormalFitter

if __name__ == "__main__":

    inspect_model(False)

    df = pd.read_parquet("data/processed/survival_data_06136.parquet")

    print(df.columns)

    df['06136_FT_observed'] = df['06136_FT_observed'].astype(bool)

    # Ensure the directory exists before saving the plot
    output_dir = os.path.join('outputs', 'plots', 'inspect_model')
    os.makedirs(output_dir, exist_ok=True)

    print(df['06136_FT_observed'].value_counts())
    print(df['06136_FT_duration'].describe())
    event_rows = df[df['06136_FT_observed'] == 1]
    print(f"\nFound {len(event_rows)} events")
    if len(event_rows) > 0:
        print("\nFirst 5 events:")
        print(event_rows.head())
    else:
        print("No events found! All observations are censored.")


    # Check for issues in the duration data
    plt.figure(figsize=(10, 6))
    plt.hist(df['06136_FT_duration'], bins=50) 
    plt.title("Distribution of Duration Values") 
    plt.savefig('duration_FTst.png')

    plt.figure()
    plt.plot(df['06136_WOG_FT'])
    plt.savefig('outputs/plots/inspect_model/duration_hist.png')
    plt.figure()


    df2 = pd.read_parquet("data/raw/precipitation_imputed_data.parquet")
    df2 = df2.clip(lower=0, upper=60)
    print(df2['06136'].isnull().sum())
    print(len(df2['06136']))
    print(len(df))
    #inspect_model()


    # DIAGNOSTIC SECTION
    print("\n=== DIAGNOSTIC INFORMATION ===")
    event_rate = df['06136_FT_observed'].mean()
    print(f"Event rate: {event_rate:.4f} ({event_rate*100:.2f}%)")

