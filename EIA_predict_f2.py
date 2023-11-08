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

from EIA_custom_transformers import HourofDay, DayofWeek, MonthofYear, LagTransformer
        
def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

def predict_f2(input_json):
    # Convert the input json into a DataFrame
    # The input data should contain the past 5 hours of demand data in the 'demand_MW' column
    input_df = pd.DataFrame(input_json['data'])
    input_df['date_time'] = pd.to_datetime(input_df['date_time'])
    input_df['demand_MW'] = input_df['demand_MW'].astype(float)
    
    # Use the fitted season models to predict base demand for the current and future hours
    input_df['base_demand_f0'] = np.expm1(season_model_f0.predict(input_df))
    base_demand_f2 = np.expm1(season_model_f2.predict(input_df))
    
    # Calculate the demand anomaly for the current hour and 2 hours in the future
    input_df['demand_anomaly_f0'] = input_df['demand_MW'] - input_df['base_demand_f0']
    demand_anomaly_f2 = residual_model_f2.predict(input_df)
    
    # Calculate the final demand forecast for 2 hours in the future
    pred_demand_f2 = base_demand_f2[-1] + demand_anomaly_f2[-1]
    
    pred_time = input_df.iloc[-1]['date_time'] + pd.DateOffset(hours=2)  
    
    pred_timestring = pred_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return pred_timestring, int(pred_demand_f2)

# Load the pickled models
season_model_f0 = load('base_nowcast.bin')
season_model_f2 = load('base_forecast_f2.bin')
residual_model_f2 = load('residual_forecast_f2.bin')

# Load 5 hours of historical demand data into the input data variables
records ={
    'data': [
        {'date_time': '2019-07-02 00:00:00', 'demand_MW': 17918},
        {'date_time': '2019-07-02 01:00:00', 'demand_MW': 17400},
        {'date_time': '2019-07-02 02:00:00', 'demand_MW': 16700},
        {'date_time': '2019-07-02 03:00:00', 'demand_MW': 15190},
        {'date_time': '2019-07-02 04:00:00', 'demand_MW': 13623}
    ]
}

# Make a demand forecast for 2 hours in the future
time_f2, forecast = predict_f2(records)
    
print(f"Forecasted demand for {time_f2}: {forecast} MW")
