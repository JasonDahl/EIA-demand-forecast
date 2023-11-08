import datetime as dt
import numpy as np
import pandas as pd

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