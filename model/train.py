# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:19:34 2022

@author: nkayf
"""
import pandas as pd
import numpy as np
import os
from modules import ExploratoryDataAnalysis, ModelCreation, ModelTraining, ModelEvaluation
from sklearn.model_selection import train_test_split

DATA_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'train.csv'))
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
LOG_PATH = os.path.join(os.getcwd(),'log')

#%% Step 1) Data Loading

df = pd.read_csv(DATA_PATH)

#%% Step 2) Data Interpretation/Inspection

df.info()
df.describe().T
df.isna().sum() 
df.boxplot() 
df[df.duplicated()]

# =============================================================================
# - check missing value : Identified for Ever_Married,Graduated,Profession,Work_Experience,Family_Size,Var_1
# - check datatype : Identified for Gender,Ever_Married,Graduated,Profession,Spending_Score,Var_1,Segmentation
# - check outliers : Identified for columns ID
# - check duplicate: No Duplicates
# =============================================================================

#%% Step 3) Data Cleaning

# Remove Outliers (column ID)

y = df.iloc[:,-1]
df = df.iloc[:,1:]
df = df.iloc[:,:-1] 

eda = ExploratoryDataAnalysis()

# Convert to numeric
df = eda.label_encode(df)
# Impute NaN 
df = eda.impute_data(df)

# =============================================================================
# - remove column ID
# - impute NaN with k-means
# - convert data and standardise datatype using label encoder
# =============================================================================

#%% Step 4) Feature Selection

eda.feature_selection(df)

# =============================================================================
# No features are selected since the highest correlation scores less than 70%
# =============================================================================

#%% Step 5) Data Preprocessing

#Encode
y = eda.one_hot_encoder(y)

# Scale
X = df
X = eda.scale_data(X)
y = eda.scale_data(y)

# =============================================================================
# Scale all features using minmax because data contains no negative values
# Encode label using OHE
# =============================================================================

#%% Step 6) Model Building

mc = ModelCreation()
model = mc.model_create(y.shape[1], X.shape[1],nb_nodes=256,dropout=.2)

# =============================================================================
# Dense = 256, Dropout = .3, Hidden Layer = 3
# =============================================================================

#%% Step 7) Model Training

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

mt = ModelTraining()
hist = mt.model_training(model, X_train,y_train, (X_test,y_test),epochs=100)
print(hist.history.keys())

# =============================================================================
# Epochs = 100 
# =============================================================================

#%% Step 8) Model Performance

mt.training_history(hist)


#%% Step 9) Model Evaluation

me = ModelEvaluation()
me.report_metrics(model,y_test,X_test)

# =============================================================================
# Not good
# =============================================================================

#%% Save Model
model.save(MODEL_PATH)


