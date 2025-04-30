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
        if train:
            print("Training the model...")
            if train:
                
                # Fix: Pass string filename first, not the model object
                station_file = "data/processed/survival_data_05005.parquet"
                station_id = "05005"
                station, station_models, timing = process_station_file(f"survival_data_{station_id}.parquet", 
                                                                    os.path.dirname(station_file), 
                                                                    False)
                
                # Only try to add models if they were successfully created
                if station_models:
                    for model_key, survival_model in station_models.items():
                        model.models[model_key] = survival_model
                    model.stations.append(station)
                    
                    # Update available soil types
                    for model_key in station_models:
                        soil_type = model_key.split('_')[1]
                        if soil_type not in model.available_soil_types:
                            model.available_soil_types.append(soil_type)
            else:
                print(f"No models were created for station {station_id}")
        else:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return
            model.load(path=model_path)
        
        # Inspect specific station
        inspect_station(model, "05005")
        
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
                                flood_prob = (1-surv_prob)*100 if isinstance(surv_prob, (int, float)) else None
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
        {"max": 1, "label": "1 Year", "filename": "1year"},
        {"max": 5, "label": "5 Years", "filename": "5years"},
        {"max": 10, "label": "10 Years", "filename": "10years"}
    ]
    
    # For each time range, create a separate plot
    for time_range in time_ranges:
        plt.figure(figsize=(10, 6))
        
        # Generate time points (convert to days for x-axis display)
        t_max = time_range["max"]  # in years
        t = np.linspace(0.01, t_max, 100)  # 100 points from 0.01 to max years (avoid 0)
        t_days = t * 365.25  # convert to days for display
        
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
                    flood_probs = [1 - prob for prob in survival_probs]
                    
                    # Plot the flood probability curve
                    plt.plot(t_days, flood_probs, label=f"Soil: {soil_type}", linewidth=2)
                    soil_types_plotted += 1
            except Exception as e:
                print(f"  Error plotting model for soil type {soil_type}: {e}")
        
        if soil_types_plotted > 0:
            plt.xlabel('Time (days)')
            plt.ylabel('Flood Probability')
            plt.title(f'Flood Probability Curves for Station {station_id} ({time_range["label"]})')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
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
from sksurv.nonparametric import kaplan_meier_estimator

if __name__ == "__main__":

    #inspect_model(True)

    df = pd.read_parquet("data/processed/survival_data_05109.parquet")

    print(df.columns)

    km = KaplanMeierFitter()
    
    km.fit(durations=df['05109_HI_TTE'],event_observed=df['05109_HI_observed'])
    
    df['05109_HI_observed'] = df['05109_HI_observed'].astype(bool)

    time, survival_prob, conf_int = kaplan_meier_estimator(df['05109_HI_observed'], df['05109_HI_duration'], conf_type="log-log")

    # Ensure the directory exists before saving the plot
    output_dir = os.path.join('outputs', 'plots', 'inspect_model')
    os.makedirs(output_dir, exist_ok=True)

    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.savefig(os.path.join(output_dir, "km_plot.png"))

    print(df['05109_HI_observed'].value_counts())
    print(df['05109_HI_duration'].describe())
    event_rows = df[df['05109_HI_observed'] == 1]
    print(f"\nFound {len(event_rows)} events")
    if len(event_rows) > 0:
        print("\nFirst 5 events:")
        print(event_rows.head())
    else:
        print("No events found! All observations are censored.")


    # Try plotting the cumulative hazard (might show the pattern better)
    plt.figure(figsize=(10, 6))
    km.plot_cumulative_density()
    plt.grid(True)
    plt.title("Cumulative density")
    plt.savefig('outputs/plots/inspect_model/cumulative_density.png')

    # Check for issues in the duration data
    plt.figure(figsize=(10, 6))
    plt.hist(df['05109_HI_duration'], bins=50) 
    plt.title("Distribution of Duration Values") 
    plt.savefig('outputs/plots/inspect_model/duration_hist.png')

    plt.figure()
    plt.plot(df['05109_WOG_HI'])
    plt.savefig("outputs/plots/inspect_model/ss21.png")

    plt.figure()


    df2 = pd.read_parquet("data/raw/precipitation_imputed_data.parquet")
    df2 = df2.clip(lower=0, upper=60)
    print(df2['05109'].isnull().sum())
    print(len(df2['05109']))
    print(len(df))
    #inspect_model()



    # DIAGNOSTIC SECTION
    print("\n=== DIAGNOSTIC INFORMATION ===")
    event_rate = df['05109_HI_observed'].mean()
    print(f"Event rate: {event_rate:.4f} ({event_rate*100:.2f}%)")

    # SOLUTION 1: Try plotting with CONSISTENT variables
    plt.figure(figsize=(10, 6))
    km_tte = KaplanMeierFitter()
    km_tte.fit(durations=df['05109_HI_TTE'], event_observed=df['05109_HI_observed'])
    km_tte.plot_cumulative_density()
    plt.grid(True)
    plt.title("Cumulative Incidence (using TTE values)")
    plt.savefig('outputs/plots/inspect_model/cumulative_density_tte.png')

    # SOLUTION 2: Try duration with events correctly marked
    plt.figure(figsize=(10, 6))
    km_dur = KaplanMeierFitter()
    km_dur.fit(durations=df['05109_HI_duration'], event_observed=df['05109_HI_observed'])
    km_dur.plot_cumulative_density()
    plt.grid(True)
    plt.title("Cumulative Incidence (using duration values)")
    plt.savefig('outputs/plots/inspect_model/cumulative_density_duration.png')


    # SOLUTION 4: Check for time window issues
    evenHI_by_time = df['05109_HI_observed'].rolling(window=1000).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(evenHI_by_time)
    plt.title("Event Rate Over Time (Moving Average)")
    plt.savefig('outputs/plots/inspect_model/event_rate_time.png')

    # Create sksurv-compatible structured array
    y = np.zeros(len(df), dtype=[('event', bool), ('time', float)])
    y['event'] = df['05109_HI_observed'].values
    y['time'] = df['05109_HI_duration'].values

    print("\nEvent time analysis:")
    event_durations = df[df['05109_HI_observed'] == 1]['05109_HI_duration'].describe()
    print(f"Event durations: {event_durations}")
    print(f"Max duration overall: {df['05109_HI_duration'].max()}")
    print(f"Events at max duration: {sum((df['05109_HI_observed'] == 1) & (df['05109_HI_duration'] == df['05109_HI_duration'].max()))}")
    

    test = WeibullFitter()
    test.fit(df['05109_HI_duration'], df['05109_HI_observed'])
    plt.figure(figsize=(10, 6))
    test.plot_cumulative_density()
    plt.grid(True)
    plt.title("Cumulative Incidence (Weibull)")
    plt.savefig(os.path.join('outputs', 'plots', 'inspect_model', 'cumulative_density_weibull.png'))
    print(f"Weibull parameters: {test.lambda_}, {test.rho_}")
    print(f"Weibull median survival time: {test.median_survival_time_}")
    print(f"Weibull AIC: {test.AIC_}")
    print(f"Weibull BIC: {test.BIC_}")

    test = ExponentialFitter()
    test.fit(df['05109_HI_duration'], df['05109_HI_observed'])
    plt.figure(figsize=(10, 6))
    test.plot_cumulative_density()
    plt.grid(True)
    plt.title("Cumulative Incidence (Exponential)")
    plt.savefig(os.path.join('outputs', 'plots', 'inspect_model', 'cumulative_density_exponential.png'))
    print(f"Exponential parameters: {test.lambda_}")
    print(f"Exponential median survival time: {test.median_survival_time_}")
    print(f"Exponential AIC: {test.AIC_}")
    print(f"Exponential BIC: {test.BIC_}")

    test = LogNormalFitter()
    test.fit(df['05109_HI_duration'], df['05109_HI_observed'])
    plt.figure(figsize=(10, 6))
    test.plot_cumulative_density()
    plt.grid(True)
    plt.title("Cumulative Incidence (LogNormal)")
    plt.savefig(os.path.join('outputs', 'plots', 'inspect_model', 'cumulative_density_lognormal.png'))
    print(f"LogNormal parameters: {test.mu_}, {test.sigma_}")
    print(f"LogNormal median survival time: {test.median_survival_time_}")
    print(f"LogNormal AIC: {test.AIC_}")
    print(f"LogNormal BIC: {test.BIC_}")
