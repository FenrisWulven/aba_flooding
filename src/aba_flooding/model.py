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

    def predict_flood_risk(self, geo_data, years=1):
        """
        Predict flood risk for geographic data.
        
        Parameters:
        -----------
        geo_data : GeoDataFrame
            Geographic data to predict flood risk for
        years : int or float
            Number of years to predict ahead
            
        Returns:
        --------
        array : Flood risk probabilities for each area
        """
        # HEre is the actual implementation later
        # TODO
        pass
