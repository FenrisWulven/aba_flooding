import pandas as pd
from lifelines import KaplanMeierFitter
import geopandas
import numpy as np
import matplotlib.pyplot as plt

# Initially a KaplanMaier model. THe idea is to have 1 for each soiltype butwe might need to cpompromise as it does not handle non flooding well.

class SurvivalModel:
    def __init__(self, soil_type='clay'):
        self.model = KaplanMeierFitter()
        self.soil_type = soil_type
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
            
        print(f"Soil type: {self.soil_type}, Year: {year}, Duration: {duration}, Max observed: {max_observed}")
        
        # Get probability and apply a dampening function to avoid 100% predictions
        # as years increase
        raw_prob = float(self.predict_proba(duration))
        
        # Apply a dampening function for multi-year predictions
        if year > 1:
            # Dampened probability that approaches but never quite reaches 100%
            prob = raw_prob * (1 - 0.1 / year)
            print(f"Year {year}: Raw prob {raw_prob:.4f}, dampened to {prob:.4f}")
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
                    print(f"Training model for {soil_type} using provided survival dataframe")
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
                
                print(f"Looking for columns: {column_duration} and {column_event}")
                
                if column_duration in data.columns and column_event in data.columns:
                    print(f"Training model for {soil_type} using columns {column_duration} and {column_event}")
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
                print(f"Creating default model for soil type '{soil_type}'")
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
                
                print(f"Soil {soil_type}, Year {year}: Prediction = {raw_prediction:.4f}")
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
            print(f"No predictions for soil types: {missing_soil_types}")
            # Set default values that increase with year but never reach 100%
            default_value = min(0.2 + 0.05 * year, 0.7)
            geodata[column_name].fillna(default_value, inplace=True)
            geodata[f'{column_name}_raw'].fillna(default_value, inplace=True)
        
        # Convert to percentage for visualization
        geodata[column_name] = geodata[column_name] * 100
        
        return geodata
    
    def plot_all(self, save=False):
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
                plt.savefig(f"{soil_type_safe}_survival_function.png")
            else:
                plt.show()