# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:07:24 2022

@author: nkayf
"""

#%% Imports and Paths

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import os

#TEST_DATA_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'new_customer.csv'))
DATA_PATH = (os.path.join(os.path.dirname(__file__), '..','data', 'train.csv'))
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
LOG_PATH = os.path.join(os.getcwd(),'log')

#%% Classes and Function

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def label_encode(self, data):
        # Convert to numeric
        le = LabelEncoder()
        df_temp = data.astype("object").apply(le.fit_transform)
        data = df_temp.where(~data.isna(), data)
        return data
    
    def impute_data(self,data):
        imputer = KNNImputer(n_neighbors=5, metric="nan_euclidean") 
        data = imputer.fit_transform(data) 
        data = pd.DataFrame(data)
        return data
    
    def feature_selection(self,data):         
        plt.figure()
        sns.heatmap(data.corr(), annot=True, cmap=plt.cm.Reds)
        plt.show()
        
    def scale_data(self,data):
        mms_scaler = MinMaxScaler()
        return mms_scaler.fit_transform(data)
                
class ModelCreation():
    def model_create(self,nb_class, input_data_shape,nb_nodes=32, activation='relu',dropout=.3):
        
        '''
        This function creates a model with 2 hidden layer
        Last layer of the model comprises of softmax activation function.
        
        Parameters
        ----------
        nb_class : Int
            DESCRIPTION.
        input_data_shape : Array
            DESCRIPTION.
        nb_nodes : Int, optional
            DESCRIPTION. The default is 32.
        activation : Array, optional
            DESCRIPTION. The default is 'relu'.
        droput : Float, optional
            DESCRIPTION. The default is 0.3
    
        Returns
        -------
        model : TYPE
            DESCRIPTION.
    
        '''
        model = Sequential()
        model.add(Input(shape=input_data_shape))
        model.add(Dense(nb_nodes,activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(nb_nodes,activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(nb_class, activation='softmax'))
        model.summary()
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy',
                      metrics='acc')    
        model.summary()
        return model
    
class ModelTraining():
    def model_training(self,model, x_train,y_train, validation_data,epochs=100):
        log_files = os.path.join(LOG_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss',patience=3)
        
        return model.fit(x_train,y_train, epochs=epochs, validation_data=validation_data,callbacks=[tensorboard_callback,early_stopping_callback])

    
    def training_history(self,hist):
        '''
        This function display
    
    
        Parameters
        ----------
        hist : Array
            Contains Feature.
    
        Returns
        -------
        None.
    
        '''
        keys = [i for i in hist.history.keys()]        
        training_loss = hist.history[keys[0]]
        training_metrics = hist.history[keys[1]]
        validation_loss = hist.history[keys[2]]
        validation_metrics = hist.history[keys[3]]
        
        # Visualise training process - matplotlib alt
        plt.figure()
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.title('training {} and validation {}'.format(keys[0], keys[0]))
        plt.xlabel('epoch')
        plt.ylabel(keys[0])
        plt.legend(['training loss','validation loss'])
        plt.show()
            
        plt.figure()
        plt.plot(training_metrics)
        plt.plot(validation_metrics)
        plt.title('training_accuracy')
        plt.xlabel('epoch')
        plt.ylabel(keys[1])
        plt.legend(['training_accuracy','epoch'])
        plt.show()
        
class ModelEvaluation():
    def report_metrics(self,model,y_test,X_test):
        model.evaluate(X_test,y_test, batch_size=100)    
        pred_x = model.predict(X_test) 
        y_true = np.argmax(y_test,axis=1)
        y_pred = np.argmax(pred_x,axis=1)
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(accuracy_score(y_true,y_pred))
        
    

   