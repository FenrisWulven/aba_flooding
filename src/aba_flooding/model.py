import pandas as pd
from lifelines import KaplanMeierFitter
import geopandas
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

# TODO: STATIONS!

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


class FloodModel:
    def __init__(self):
        self.models = {}
        self.is_fitted = False
        self.units = 'hours'
        self.soil_types = ["DG - Meltwater gravel", "DS - Meltwater sand"]
        self.stations = []
        self.data = None
        self.available_soil_types = []
    
    def train(self, data, survival_dfs=None, duration_column='duration', event_column='observed'):
        """
        Train the survival models for each soil type.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Original processed dataframe (can be None if survival_dfs is provided)
        survival_dfs : dict, optional
            Dictionary mapping soil types to their survival dataframes
        duration_column : str
            Name of duration column in each survival df
        event_column : str
            Name of event observation column in each survival df
        """
        self.data = data
        self.available_soil_types = []
        
        # If survival dataframes are provided, use them directly
        if survival_dfs and isinstance(survival_dfs, dict):
            for soil_type, survival_df in survival_dfs.items():
                if len(survival_df) > 0:
                    #print(f"Training model for {soil_type} using provided survival dataframe")
                    model = SurvivalModel(soil_type=soil_type)
                    model.train(survival_df, duration_column, event_column)
                    self.models[soil_type] = model
                    self.available_soil_types.append(soil_type)
                else:
                    print(f"Warning: Empty survival dataframe for soil type '{soil_type}'")
        
        # Legacy approach using the original data format
        elif data is not None:
            for soil_type in self.soil_types:
                # Correctly form the full column names
                column_duration = f"{soil_type}{duration_column}"
                column_event = f"{soil_type}{event_column}"
                
                #print(f"Looking for columns: {column_duration} and {column_event}")
                
                if column_duration in data.columns and column_event in data.columns:
                    #print(f"Training model for {soil_type} using columns {column_duration} and {column_event}")
                    # Create subset with only the needed columns
                    subset_df = data[[column_duration, column_event]].rename(
                        columns={column_duration: 'duration', column_event: 'observed'})
                    
                    model = SurvivalModel(soil_type=soil_type)
                    model.train(subset_df, 'duration', 'observed')
                    self.models[soil_type] = model
                    self.available_soil_types.append(soil_type)
                else:
                    print(f"Warning: No training data found for soil type '{soil_type}'")
        else:
            raise ValueError("Either data or survival_dfs must be provided")
            
        # Handle any missing soil types with default models
        for soil_type in self.soil_types:
            if soil_type not in self.models:
                #print(f"Creating default model for soil type '{soil_type}'")
                default_model = SurvivalModel(soil_type=soil_type)
                # Use an existing model if available
                if len(self.models) > 0:
                    some_model = next(iter(self.models.values()))
                    self.models[soil_type] = some_model
                else:
                    self.models[soil_type] = default_model

        self.is_fitted = True
        return self
    
    def predict_proba(self, geodata, year):
        """
        Predict probability of rain occurring by the given year for each soil type.
        
        Parameters:
        -----------
        geodata : GeoDataFrame
            GeoDataFrame containing soil types and their geometries
        year : int
            Year to predict probability for
            
        Returns:
        --------
        GeoDataFrame : Original geodata with added predictions column
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
            
        predictions = {}
        prediction_stats = []
        
        # Iterate through each soil type in our models
        for soil_type, model in self.models.items():
            try:
                # Get the raw prediction
                raw_prediction = model.predict(year)
                
                # Store raw prediction for debugging
                predictions[soil_type] = raw_prediction
                prediction_stats.append({
                    'soil_type': soil_type,
                    'year': year,
                    'prediction': raw_prediction
                })
                
                #print(f"Soil {soil_type}, Year {year}: Prediction = {raw_prediction:.4f}")
            except Exception as e:
                print(f"Error predicting for soil type {soil_type}: {str(e)}")
                predictions[soil_type] = 0.5  # Default value on error
        
        # Print summary statistics
        if prediction_stats:
            pred_values = [p['prediction'] for p in prediction_stats]
            if pred_values:
                print(f"Year {year} prediction stats: min={min(pred_values):.4f}, "
                      f"max={max(pred_values):.4f}, avg={sum(pred_values)/len(pred_values):.4f}")
        
        # Add predictions to the GeoDataFrame
        column_name = f'predictions_{year}'
        geodata[column_name] = geodata['sediment'].map(predictions)
        
        # Store raw probability values before percentage conversion
        geodata[f'{column_name}_raw'] = geodata[column_name].copy()
        
        # Handle missing soil types
        missing_soil_types = geodata.loc[geodata[column_name].isna(), 'sediment'].unique()
        if len(missing_soil_types) > 0:
            #print(f"No predictions for soil types: {missing_soil_types}")
            # Set default values that increase with year but never reach 100%
            default_value = min(0.2 + 0.05 * year, 0.7)
            geodata[column_name].fillna(default_value, inplace=True)
            geodata[f'{column_name}_raw'].fillna(default_value, inplace=True)
        
        # Convert to percentage for visualization
        geodata[column_name] = geodata[column_name] * 100
        
        return geodata
    
    def plot_all(self, save=False, output_dir='reports/figures/'):
        """Plot survival functions for all soil types."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before plotting")
        
        for soil_type, model in self.models.items():
            plt.figure(figsize=(10, 6))
            model.plot()
            plt.title(f"Survival Function for {soil_type}")
            plt.xlabel("Duration (hours)")
            plt.ylabel("Survival Probability")
            plt.grid()
            if save:
                # make the soil type safe for saving by changing '/' to '_'
                soil_type_safe = soil_type.replace('/', '_')
                plt.savefig(f"{output_dir}{soil_type_safe}_survival_function.png")
            else:
                plt.show()
    
    def load(path):
        # Takes a pickle object and loads it
        pass