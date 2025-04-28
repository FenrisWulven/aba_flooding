import pandas as pd
from lifelines import KaplanMeierFitter
# import geopandas
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import joblib

class SurivalDeepCoxModel:
    """
    Deep Cox Proportional Hazards model wrapper using pycox and torchtuples.
    """
    def __init__(self, input_cols=None, hidden_layers=(32, 32), dropout=0.1,
                 optimizer=torch.optim.Adam, device=None):
        """
        Parameters:
        -----------
        input_cols : list of str
            List of column names to use as features.
        hidden_layers : tuple of int
            Sizes of hidden layers in the MLP.
        dropout : float
            Dropout rate.
        optimizer : torch optimizer class
            Optimizer for training.
        device : torch.device or str
            Device to train on ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.input_cols = input_cols or []
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.optimizer_cls = optimizer
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Placeholders
        self.mapper = None
        self.scaler = None
        self.net = None
        self.model = None
        self.baseline_hazards_ = None
        self.is_fitted = False

    def _build_network(self, in_features):
        return tt.practical.MLPVanilla(in_features, self.hidden_layers, 1, dropout=self.dropout)

    def train(self, df, duration_col='TTE', event_col='heavy_rain', epochs=10, batch_size=32, verbose=True):
        """
        Train the Deep Cox PH model.
        """
        if df is None or df.empty:
            raise ValueError("Training dataframe is empty.")
        # Extract features, durations, and events
        x_raw = df[self.input_cols].values.astype('float32')
        durations = df[duration_col].values.astype('float32')
        events = df[event_col].astype(int).values

        # Scale inputs
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x_raw)

        # Build network
        in_features = x.shape[1]
        self.net = self._build_network(in_features).to(self.device)

        # CoxPH model
        self.model = CoxPH(self.net, optimizer=self.optimizer_cls)

        # Fit model
        self.model.fit(x, (durations, events), epochs=epochs, batch_size=batch_size, verbose=verbose)

        # Compute baseline hazards
        self.model.compute_baseline_hazards()
        self.baseline_hazards_ = self.model.baseline_cumulative_hazards_

        self.is_fitted = True
        return self

    def predict_surv_df(self, df_new):
        """
        Predict survival function DataFrame for new samples.

        Returns a DataFrame indexed by time, with one column per sample.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained or loaded before prediction.")
        x_raw = df_new[self.input_cols].values.astype('float32')
        x = self.scaler.transform(x_raw)
        surv = self.model.predict_surv_df(x)
        return surv

    def predict_proba(self, df_new, times=None):
        """
        Predict event probability up to specified times for new data.

        Parameters:
        -----------
        df_new : DataFrame of new samples
        times : array-like
            Time points at which to compute event probability.

        Returns:
        --------
        np.ndarray : probabilities of event by each time for each sample.
        """
        surv = self.predict_surv_df(df_new)
        if times is not None:
            surv = surv.loc[times]
        # event probability = 1 - survival
        return 1 - surv.values

    def predict_median(self, df_new):
        """
        Predict median survival time for each new sample.
        """
        surv = self.predict_surv_df(df_new)
        # median survival: time when survival crosses 0.5
        medians = surv.apply(lambda col: col[col <= 0.5].index[0] if any(col <= 0.5) else surv.index.max(), axis=0)
        return medians.values

    def plot_baseline(self, save_path=None):
        """
        Plot and optionally save the baseline survival curve.
        """
        if self.baseline_hazards_ is None:
            raise RuntimeError("Baseline hazards not computed. Fit the model first.")
        H0 = self.baseline_hazards_
        S0 = np.exp(-H0)
        plt.figure(figsize=(8, 5))
        plt.step(S0.index, S0.values, where='post')
        plt.title("Baseline Survival Curve")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

    def save(self, path):
        """
        Save model state and scaler to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # save torch network
        torch.save(self.net.state_dict(), path)
        # save scaler
        scaler_path = path + '.scaler.pkl'
        pd.to_pickle(self.scaler, scaler_path)

    def load(self, path):
        """
        Load model state and scaler from disk.
        """
        # load scaler
        scaler_path = path + '.scaler.pkl'
        self.scaler = pd.read_pickle(scaler_path)
        # rebuild network
        in_features = len(self.input_cols)
        self.net = self._build_network(in_features).to(self.device)
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()
        # rebuild model wrapper
        self.model = CoxPH(self.net, optimizer=self.optimizer_cls)
        self.model.compute_baseline_hazards()
        self.baseline_hazards_ = self.model.baseline_cumulative_hazards_
        self.is_fitted = True
        return self

    def evaluate(self, df, duration_col='TTE', event_col='heavy_rain'):
        """
        Evaluate concordance index on a dataset.
        """
        durations = df[duration_col].values.astype('float32')
        events = df[event_col].astype(int).values
        surv = self.predict_surv_df(df)
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        return ev.concordance_td()

    def plot_sample_curves(self, df_new, n_samples=5, save_path=None):
        """
        Plot individual survival curves for first n_samples.
        """
        surv = self.predict_surv_df(df_new)
        times = surv.index
        plt.figure(figsize=(10, 6))
        for i in range(min(n_samples, surv.shape[1])):
            plt.step(times, surv.iloc[:, i], where='post', label=f"Sample {i+1}")
        plt.title("Survival Curves")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.grid(True)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

class SurvivalModel:
    def __init__(self, soil_type='clay'):
        self.model = KaplanMeierFitter()
        self.soil_type = soil_type
        self.station = None  # Placeholder for station data
        self.is_fitted = False
        self.units = 'hours'  # Default unit for duration 
    
    def train(self, df, duration_column='duration', event_column='observed'):
        """Train the survival model on dry spell durations."""
        if df is None or len(df) == 0:
            raise ValueError("Training data is empty")
            
        self.model.fit(durations=df[duration_column], event_observed=df[event_column])
        self.is_fitted = True
        return self
    
    def predict_proba(self, durations):
        """
        Predict probability of rain occurring by the given duration.
        
        Parameters:
        -----------
        durations : int, array-like
            Number of time units to predict probability for
            
        Returns:
        --------
        array-like : Probability of rain occurring by the specified duration
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
            
        # Convert to array if single value
        if isinstance(durations, (int, float)):
            durations = [durations]
            
        # Get survival function (probability of remaining dry)
        survival_probs = self.model.predict(durations)
        
        # Return probability of rain (1 - survival probability)
        return 1 - survival_probs
    
    def predict(self, year):
        """
        Predict the probability of rain occurring before a given year (int)
        
        Parameters:
        -----------
        year : int
            Number of years into the future
            
        Returns:
        --------
        float : Probability of rain occurring by the specified year
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get the maximum observed duration from the model
        max_observed = self.model.timeline.max()
        
        # Calculate duration based on the year
        if self.units == 'hours':
            # Scale down for better distribution across years
            # This prevents all predictions from being 100%
            if year <= 0.1:  # For very short periods
                duration = year * 365 * 24 * 0.1  # Scale down for short durations
            elif year <= 1:  # For periods up to a year
                duration = year * 365 * 24 * 0.2  # Scale down a bit
            else:  # For longer periods
                # Use a logarithmic scale to prevent saturation at 100%
                duration = min(max_observed * (1 + np.log(year)), max_observed)
        elif self.units == 'days':
            if year <= 1:
                duration = year * 365 * 0.5
            else:
                duration = min(max_observed * (1 + np.log(year)), max_observed)
        else:  # years or other units
            duration = min(year, max_observed)
            
        #print(f"Soil type: {self.soil_type}, Year: {year}, Duration: {duration}, Max observed: {max_observed}")
        
        # Get probability and apply a dampening function to avoid 100% predictions
        # as years increase
        raw_prob = float(self.predict_proba(duration))
        
        # Apply a dampening function for multi-year predictions
        if year > 1:
            # Dampened probability that approaches but never quite reaches 100%
            prob = raw_prob * (1 - 0.1 / year)
            #print(f"Year {year}: Raw prob {raw_prob:.4f}, dampened to {prob:.4f}")
        else:
            prob = raw_prob
        
        return prob
    
    def plot(self):
        """Plot the survival function."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before plotting")
        
        plt.figure(figsize=(10, 6))
        self.model.plot_cumulative_density()
        plt.title(f"Survival Function for {self.soil_type}")
        plt.xlabel("Duration (hours)")
        plt.ylabel("Survival Probability")
        plt.grid()
        plt.savefig(f"{self.soil_type}_survival_function.png")

    def save(self, path):
        """
        Save the fitted Kaplan-Meier model to disk.
        
        Parameters:
        -----------
        path : str
            File path where the model should be saved
        
        Returns:
        --------
        self : SurvivalModel
            Returns self for method chaining
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'soil_type': self.soil_type,
            'station': self.station,
            'units': self.units,
            'is_fitted': self.is_fitted
        }
        pd.to_pickle(model_data, path)
        return self
    
    def load(self, path):
        """
        Load a fitted Kaplan-Meier model from disk.
        
        Parameters:
        -----------
        path : str
            File path to the saved model
            
        Returns:
        --------
        self : SurvivalModel
            Returns self with loaded model
        """
        # Load model and metadata
        model_data = pd.read_pickle(path)
        
        # Restore model attributes
        self.model = model_data['model']
        self.soil_type = model_data['soil_type']
        self.station = model_data['station']
        self.is_fitted = model_data['is_fitted']
        self.units = model_data['units']
        
        return self

class FloodModel:
    def __init__(self):
        self.models = {}
        self.is_fitted = False
        self.units = 'hours'
        self.soil_types = ["DG - Meltwater gravel", "DS - Meltwater sand"]
        self.stations = []
        self.available_soil_types = []
    
    def add_station(self, station, survival_df, soiltypes):
        """
        Add station data to the flood model and train survival models for each soil type.
        
        Parameters:
        -----------
        station : str
            Station identifier
        survival_df : pandas.DataFrame
            DataFrame containing survival data for the station
        soiltypes : list
            List of soil types to train models for this station
            
        Returns:
        --------
        self : FloodModel
            Returns self for method chaining
        """
        if station not in self.stations:
            self.stations.append(station)
            
        # Create models for each soil type in this station
        for soil_type in soiltypes:
            # Create column names based on pattern in the dataframe
            duration_column = f"{station}_{soil_type}_duration"
            event_column = f"{station}_{soil_type}_observed"
            
            # Check if needed columns exist
            if duration_column in survival_df.columns and event_column in survival_df.columns:
                # Filter out any missing values
                valid_data = survival_df[[duration_column, event_column]].dropna()
                
                if len(valid_data) > 0:
                    # Create a model for this station-soil combination
                    model_key = f"{station}_{soil_type}"
                    
                    # Create and train the model
                    model = SurvivalModel(soil_type=soil_type)
                    model.station = station
                    model.train(
                        valid_data.rename(columns={
                            duration_column: 'duration',
                            event_column: 'observed'
                        }),
                        'duration', 
                        'observed'
                    )
                    
                    # Add to our models dictionary
                    self.models[model_key] = model
                    
                    # Add to available soil types if not already there
                    if soil_type not in self.available_soil_types:
                        self.available_soil_types.append(soil_type)
                    
                    print(f"Trained model for station {station}, soil type {soil_type} with {len(valid_data)} observations")
                else:
                    print(f"No valid data for station {station}, soil type {soil_type}")
            else:
                print(f"Missing columns for station {station}, soil type {soil_type}")
        
        # Mark as fitted if we have any models
        if self.models:
            self.is_fitted = True
            
        return self
    
    def save(self, path, split_by_station=True):
        """
        Save the FloodModel to disk.
        
        Parameters:
        -----------
        path : str
            File path where the model should be saved
        split_by_station : bool
            If True, save each station's models in a separate file
            
        Returns:
        --------
        self : FloodModel
            Returns self for method chaining
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Split storage by station
        if split_by_station and len(self.models) > 0:
            # Extract base directory and filename without extension
            base_dir = os.path.dirname(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            # Create stations directory
            stations_dir = os.path.join(base_dir, f"{base_name}_stations")
            os.makedirs(stations_dir, exist_ok=True)
            
            print(f"Starting split save: {len(self.stations)} stations with {len(self.models)} models...")
            
            # Group models by station
            station_models = {}
            for model_key, model in self.models.items():
                station = model_key.split('_')[0]
                if station not in station_models:
                    station_models[station] = {}
                station_models[station][model_key] = model
            
            # Create a metadata model that references station files
            meta_model = {
                'is_fitted': True,
                'units': self.units,
                'soil_types': self.soil_types,
                'stations': self.stations,
                'available_soil_types': self.available_soil_types,
                'station_paths': {}  # Will store paths to station model files
            }
            
            # Save each station separately
            saved_files = 0
            for station, models in station_models.items():
                station_path = os.path.join(stations_dir, f"station_{station}.joblib")
                
                # Display progress every 10 stations
                if saved_files % 10 == 0:
                    print(f"Saving station {saved_files}/{len(station_models)}: {station} with {len(models)} models...")
                    
                # Save station models with joblib
                joblib.dump(models, station_path, compress=3)
                
                # Store the relative path in metadata
                meta_model['station_paths'][station] = os.path.relpath(station_path, base_dir)
                saved_files += 1
                
            # Save the metadata file
            print(f"Saving metadata to {path}...")
            joblib.dump(meta_model, path, compress=3)
            print(f"Successfully saved {saved_files} station files and metadata")
            
        else:
            # Traditional single-file save
            print(f"Starting to save {len(self.models)} models to {path}...")
            model_data = {
                'models': self.models,
                'is_fitted': self.is_fitted,
                'units': self.units,
                'soil_types': self.soil_types,
                'stations': self.stations,
                'available_soil_types': self.available_soil_types
            }
            joblib.dump(model_data, path, compress=3)
            print(f"Successfully saved model to {path}")
            
        return self

    def load(self, path, lazy_load=True):
        """
        Load a saved FloodModel from disk.
        
        Parameters:
        -----------
        path : str
            File path to the saved model
        lazy_load : bool
            If True and model was saved with split_by_station=True, 
            only load station models when requested
            
        Returns:
        --------
        self : FloodModel
            Returns self with loaded models
        """
        print(f"Loading model from {path}...")
        
        # Try to load model data
        model_data = joblib.load(path)
        
        # Check if this is a split model (metadata file)
        if isinstance(model_data, dict) and 'station_paths' in model_data:
            # This is a split model - load metadata
            self.is_fitted = model_data.get('is_fitted', False)
            self.units = model_data.get('units', 'hours')
            self.soil_types = model_data.get('soil_types', [])
            self.stations = model_data.get('stations', [])
            self.available_soil_types = model_data.get('available_soil_types', [])
            
            # Get base directory for relative paths
            base_dir = os.path.dirname(path)
            
            if lazy_load:
                # Create a proxy function for each station that will load data when needed
                self.models = {}
                print(f"Lazy-loading enabled: Referenced {len(model_data['station_paths'])} stations")
                
                # Store the station paths for later loading
                self._station_paths = {
                    station: os.path.join(base_dir, rel_path) 
                    for station, rel_path in model_data['station_paths'].items()
                }
            else:
                # Load all station models immediately
                self.models = {}
                total_stations = len(model_data['station_paths'])
                print(f"Loading all {total_stations} station models...")
                
                for i, (station, rel_path) in enumerate(model_data['station_paths'].items()):
                    station_path = os.path.join(base_dir, rel_path)
                    if i % 10 == 0:
                        print(f"Loading station {i+1}/{total_stations}: {station}...")
                    
                    try:
                        # Load the station models
                        station_models = joblib.load(station_path)
                        # Add to the main models dictionary
                        self.models.update(station_models)
                    except Exception as e:
                        print(f"Error loading station {station}: {e}")
        else:
            # Traditional single-file model
            self.models = model_data.get('models', {})
            self.is_fitted = model_data.get('is_fitted', False)
            self.units = model_data.get('units', 'hours')
            self.soil_types = model_data.get('soil_types', [])
            self.stations = model_data.get('stations', [])
            self.available_soil_types = model_data.get('available_soil_types', [])
        
        print(f"Model loaded with {len(self.stations)} stations")
        return self

    def get_station_models(self, station):
        """
        Get all models for a specific station.
        Will load from disk if using lazy loading.
        
        Parameters:
        -----------
        station : str
            Station identifier
            
        Returns:
        --------
        dict : Dictionary of models for the station
        """
        # Check if we're using lazy loading and need to load this station
        if hasattr(self, '_station_paths') and station in self._station_paths:
            # Station not loaded yet, load it now
            station_path = self._station_paths[station]
            print(f"Loading station {station} models from {station_path}...")
            
            try:
                # Load the station models
                station_models = joblib.load(station_path)
                # Add to the main models dictionary
                self.models.update(station_models)
                # Return the loaded models for this station
                return {k: v for k, v in station_models.items()}
            except Exception as e:
                print(f"Error loading station {station}: {e}")
                return {}
        
        # If not lazy loading or already loaded, filter existing models
        return {k: v for k, v in self.models.items() if k.startswith(f"{station}_")}

    def load_station(self, station, stations_dir):
        """
        Load models for a specific station from the stations directory.
        
        Parameters:
        -----------
        station : str
            Station identifier
        stations_dir : str
            Directory containing station model files
            
        Returns:
        --------
        dict : Dictionary of loaded models for this station
        """
        # Try different filename patterns
        file_patterns = [
            os.path.join(stations_dir, f"{station}.joblib"),
            os.path.join(stations_dir, f"station_{station}.joblib")
        ]
        
        loaded_models = {}
        for file_path in file_patterns:
            if os.path.exists(file_path):
                try:
                    print(f"Loading station models from {file_path}")
                    station_models = joblib.load(file_path)
                    
                    # Add models to the main models dictionary
                    if isinstance(station_models, dict):
                        for model_key, model in station_models.items():
                            self.models[model_key] = model
                            loaded_models[model_key] = model
                    
                    # Return the loaded models
                    return loaded_models
                except Exception as e:
                    print(f"Error loading station file {file_path}: {e}")
        
        print(f"No valid model file found for station {station} in {stations_dir}")
        return {}