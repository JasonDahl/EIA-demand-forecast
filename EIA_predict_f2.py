import datetime as dt
import numpy as np
import pandas as pd

import pickle

from sklearn import base
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import ensemble

#define custom transformers
class HourofDay(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, column_name='target', horizon=0):
        self.column_name=column_name
        self.horizon = horizon
    
    def fit(self, X, y=None):
        return self
    
    def hour_vector(self, hour):
        v = np.zeros(24)
        v[hour] = 1
        return v
    
    def transform(self, X):
        # Adjust the timestamp with the horizon offset
        adjusted_timestamp = X[self.column_name] + pd.DateOffset(hours=self.horizon)
        
        # Extract the hour of the day for the adjusted timestamp
        hour_of_day = adjusted_timestamp.dt.hour
        
        return np.stack([self.hour_vector(h) for h in hour_of_day])
    
class DayofWeek(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, column_name='target', horizon=0):
        self.column_name=column_name
        self.horizon=horizon
    
    def fit(self, X, y=None):
        return self
    
    def weekday_vector(self, weekday):
        v = np.zeros(7)
        v[weekday] = 1
        return v
    
    def transform(self, X):
        # Adjust the timestamp with the horizon offset
        adjusted_timestamp = X[self.column_name] + pd.DateOffset(hours=self.horizon)
        
        # Extract the day of the week for the adjusted timestamp
        day_of_week = adjusted_timestamp.dt.dayofweek
        
        return np.stack([self.weekday_vector(d) for d in day_of_week])

class MonthofYear(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, column_name='target', horizon=0):
        self.column_name=column_name
        self.horizon=horizon
    
    def fit(self, X, y=None):
        return self
    
    def month_vector(self, month):
        v = np.zeros(12)
        v[month-1] = 1
        return v
    
    def transform(self, X):
        # Adjust the timestamp with the horizon offset
        adjusted_timestamp = X[self.column_name] + pd.DateOffset(hours=self.horizon)
        
        # Extract the hour of the day for the adjusted timestamp
        month_of_year = adjusted_timestamp.dt.month
        
        return np.stack([self.month_vector(m) for m in month_of_year])
    
class LagTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, column_name='label', lag=1):
        self.lag=lag
        self.column_name=column_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Create a copy of the input DataFrame to avoid altering the original
        X_copy = X.copy()
        
        # Shift the specified column by 'lag' rows
        if self.lag <= len(X) - 1:
            X_copy[f'{self.column_name}-{self.lag}'] = X_copy[self.column_name].shift(self.lag)
            X_copy[f'{self.column_name}-{self.lag}'] = X_copy[f'{self.column_name}-{self.lag}'].fillna(method='bfill', axis=0)
            return X_copy[[f'{self.column_name}-{self.lag}']]
        else:
            print(f'Requested shift {self.lag} > number of records. Returning {self.column_name}-{len(X_copy)-1}')
            X_copy[f'{self.column_name}-{len(X_copy)-1}'] = X_copy[self.column_name].shift(len(X_copy) - 1)
            X_copy[f'{self.column_name}-{len(X_copy)-1}'] = X_copy[f'{self.column_name}-{len(X_copy)-1}'].fillna(method='bfill', axis=0)
            return X_copy[f'{self.column_name}-{len(X_copy)-1}']
        
def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

# Define a function to make predictions
def predict_demand(input_times, input_demands):
    # Convert the input data into a DataFrame with the same structure as f0_train
    # The input data should contain the past 5 hours of demand data in the 'demand_MW' column
    time_objects = [pd.to_datetime(time_str) for time_str in input_times]

    input_df = pd.DataFrame({'date_time': time_objects, 'demand_MW': list(map(float, input_demands))})
    
    # Use the fitted season models to predict base demand for the current and future hours
    input_df['base_demand_f0'] = np.expm1(season_model_f0.predict(input_df))
    base_demand_f2 = np.expm1(season_model_f2.predict(input_df))
    
    # Calculate the demand anomaly for the current hour and 2 hours in the future
    input_df['demand_anomaly_f0'] = input_df['demand_MW'] - input_df['base_demand_f0']
    demand_anomaly_f2 = residual_model_f2.predict(input_df)
    
    # Calculate the final demand forecast for 2 hours in the future
    pred_demand_f2 = base_demand_f2[-1] + demand_anomaly_f2[-1]
    
    return int(pred_demand_f2)

# Load the pickled models
season_model_f0 = load('base_nowcast.bin')
season_model_f2 = load('base_forecast_f2.bin')
residual_model_f2 = load('residual_forecast_f2.bin')

# Load 5 hours of historical demand data into the input data variables
input_times = ['2019-07-02 00:00:00', '2019-07-02 01:00:00', '2019-07-02 02:00:00', 
               '2019-07-02 03:00:00', '2019-07-02 04:00:00']

input_demands = [17918, 17400, 16700, 15190, 13623]

# Make a demand forecast for 2 hours in the future
forecast = predict_demand(input_times, input_demands)
    
print(f"Forecasted demand for 2 hours in the future: {forecast} MW")
