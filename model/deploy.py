# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:08:48 2022

@author: nkayf
"""

from modules import ExploratoryDataAnalysis
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

RESULT_PATH = (os.path.join(os.path.dirname(__file__), '..','data', '[UPDATED]new_customers.csv'))
DATA_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'new_customers.csv'))
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
LOG_PATH = os.path.join(os.getcwd(),'log')

#%% Step 1) Data Loading

df = pd.read_csv(DATA_PATH)
column_names = list(df.columns)
model = load_model(MODEL_PATH)
       
#%% Step 3) Data Cleaning

df_temp = df.iloc[:,0].to_numpy()
df = df.iloc[:,1:]
eda = ExploratoryDataAnalysis()

# Convert to numeric
df = eda.label_encode(df)

# Impute NaN 
df = eda.impute_data(df)

#%% Step 4) Feature Selection
#%% Step 5) Data Preprocessing

df = eda.scale_data(df)

#%% Step 6) Model Predict

outcome = model.predict(df)
outcome = np.argmax(outcome, axis=1)
segment_dict = {0:'A',1:'B',2:'C',3:'D'}
outcome = [segment_dict[key] for key in outcome]

#%% Step 7) Write into csv

df = pd.DataFrame(np.concatenate((np.expand_dims(df_temp,axis=-1), df,np.expand_dims(outcome,axis=-1)), axis=1),columns=column_names)
df.to_csv(RESULT_PATH, index=False)