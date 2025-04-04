import pandas as pd
from lifelines import KaplanMeierFitter
import geopandas

import pandas as pd
from lifelines import KaplanMeierFitter
import geopandas
import numpy as np

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
        Predict the the probability of rain occuring before a given year (int)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
            
        # Get the duration in years
        if self.units == 'hours':
            duration = year * 365 * 24
        elif self.units == 'days':
            duration = year * 365
        elif self.units == 'years':
            duration = year
        else:
            raise ValueError("Unsupported unit type. Supported units are 'hours', 'days', and 'years'.")
        
        return self.predict_proba(duration)


class FloodModel:
    def __init__(self):
        self.models = {}
        self.is_fitted = False
        self.units = 'hours'
        self.soil_types = ["DG - Meltwater gravel", "DS - Meltwater sand"]
        self.data = None
    
    def train(self, df, duration_column='duration', event_column='observed'):
        """Train the survival model on dry spell durations."""
        if df is None or len(df) == 0:
            raise ValueError("Training data is empty")
            
        self.data = df
        # Keep track of which soil types we actually have data for
        self.available_soil_types = []
        
        for soil_type in self.soil_types:
            # Check if the needed columns exist for this soil type
            column_duration = soil_type + duration_column
            column_event = soil_type + event_column
            
            if column_duration in df.columns and column_event in df.columns:
                model = SurvivalModel(soil_type=soil_type)
                model.train(df, column_duration, column_event)
                self.models[soil_type] = model
                self.available_soil_types.append(soil_type)
            else:
                print(f"Warning: No training data found for soil type '{soil_type}'. Using default model.")
                # Create a default model or use a similar soil type's model
                default_model = SurvivalModel(soil_type=soil_type)
                # Assign an existing model if available, or create a simple default
                if len(self.models) > 0:
                    # Use the first available model as a fallback
                    some_model = next(iter(self.models.values()))
                    self.models[soil_type] = some_model
                else:
                    # If no models exist yet, create a simple default model
                    # This is just a placeholder - you may want to improve this
                    # by using a more appropriate default model
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
        
        # Iterate through each soil type in our models
        for soil_type, model in self.models.items():
            predictions[soil_type] = model.predict(year)
        
        # Add predictions for year to the GeoDataFrame
        geodata[f'predictions_{year}'] = geodata['sediment'].map(predictions)
        
        # Set default probability of 0.1 for soil types not in our models
        geodata[f'predictions_{year}'].fillna(0.1, inplace=True)  # Changed from 0 to 0.1
        
        # Convert predictions to percentage
        geodata[f'predictions_{year}'] *= 100
        geodata[f'predictions_{year}'] = geodata[f'predictions_{year}'].astype(int)
                
        return geodata
