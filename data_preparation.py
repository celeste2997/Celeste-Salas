# data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
from datetime import datetime
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
#from app2021 import pred_h
#import app2021
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import sys


#global pred_h
def data_processing(data_file):

    try:
        global df
        df = pd.read_csv(data_file,
                 parse_dates={'dt' : ['timestamp']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='dt')
        columns_match = df.columns
        print(set(columns_match),"Data Coulmns--------")

        columns_titles = ["weight","humidity","temperature","season"]
        print(set(columns_titles),"Required Data Columns Format-------------")

        if set(columns_titles) == set(columns_match):
            pass
        else:
            print("Enter Valid File Column format")
            sys.exit()
        

        df=df.reindex(columns=columns_titles)

    except:
        print("Enter Valid File Column format or File Name")
        sys.exit()

    df1=df.copy()

    # further resampling so that the frequency becomes daily and taking mean
    #df = df.resample('D').mean()
    dataset_train_actual = df.copy()
    dataset_train_actual = dataset_train_actual.fillna(dataset_train_actual.mean())
    print(dataset_train_actual)
    dataset_train_actual = dataset_train_actual.reset_index()
    dataset_train_timeindex = dataset_train_actual.set_index('dt')
    dataset_train = dataset_train_actual.copy()
    print(dataset_train)

    plot_cols = ['temperature', 'humidity', 'season', 'weight'] 
    plot_features = dataset_train [plot_cols]
    plot_features.index = dataset_train['dt'] 
    
    # Select features (columns) to be involved intro training and predictions
    cols = list(dataset_train)[1:5]

    # Extract dates (will be used in visualization)
    datelist_train = list(dataset_train['dt'])
    datelist_train = [date for date in datelist_train]
    #print("date list", datelist_train)
    print('Training set shape == {}'.format(dataset_train.shape))
    print('All timestamps == {}'.format(len(datelist_train)))
    print('Featured selected: {}'.format(cols))
    dataset_train = dataset_train[cols].astype(str)
    for i in cols:
        for j in range(0, len(dataset_train)):
            dataset_train[i][j] = dataset_train[i][j].replace(',', '')

    dataset_train1 = dataset_train.astype(float)

    # Using multiple features (predictors)
    training_set = dataset_train1.values

    print('Shape of training set == {}.'.format(training_set.shape))
    #first_col = df.iloc[:, :1]

    #print('Shape of training set == ', )
    # Feature Scaling


    #temp = plot_features
    #return X_train, y_train, sc_predict, n_future, n_past, dataset_train, datelist_train, dataset_train_timeindex, plot_features.index, df1
    return dataset_train1, datelist_train, dataset_train_timeindex, df1,training_set

