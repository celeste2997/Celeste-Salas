# data preprocessing
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
from datetime import datetime
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from data_preparation import data_processing



print("Data Processing Done...")
def build_model_st(dataset_train, n_past):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True, input_shape = (n_past, dataset_train.shape [1]-1))),
        #tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(70, return_sequences=False)),
        #tf.keras.layers.LSTM(50, return_sequences=False),
        # Adding Dropout,,

        tf.keras.layers.Dropout(0.1),
        #tf.keras.layers.Dense(25, activation ='relu'),
        tf.keras.layers.Dense(1, activation ='linear'),])

    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.8)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=["mse"])
    return model


def model_train(model, X_train, y_train):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    es = EarlyStopping(monitor='val_loss', mode='min',patience=30)
    history = model.fit(X_train, y_train, epochs=50,batch_size=256, validation_split=0.2,callbacks=[es, mc])
    return history

def load_best_model(m):
    # load a saved model
    saved_model = load_model(m)
    return saved_model


# summarize history for loss
def plot_loss(history):
    loss_train = np.array(history.history['mse']).mean()
    loss_val = np.array(history.history['val_mse']).mean()
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(["Train MSE: {:.4f}".format(loss_train), "Val MSE: {:.4f}".format(loss_val)], loc='upper right')
    plt.show()


def model_prediction(datelist_train, best_model, n_future, n_past, X_train, sc_predict):
    # Generate list of sequence of days for predictions
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='h').tolist()

    '''
    Remeber, we have datelist_train from begining.
    '''
    # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())
    # Perform predictions
    predictions_future = best_model.predict(X_train[-n_future:])
    predictions_train = best_model.predict(X_train[n_past:])
    y_pred_future = sc_predict.inverse_transform(predictions_future)
    y_pred_train = sc_predict.inverse_transform(predictions_train)

    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['weight']).set_index(pd.Series(datelist_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['weight']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))
    return PREDICTIONS_FUTURE, PREDICTION_TRAIN, datelist_future_

def plot_predictions(PREDICTIONS_FUTURE, PREDICTION_TRAIN, dataset_train_timeindex):
    # Set plot size 
    # from pylab import rcParams
    plt.rcParams['figure.figsize'] = 14, 5

    # Plot parameters
    START_DATE_FOR_PLOTTING = '2017-01-07'

    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['weight'], color='r', label='Predicted Weights')
    plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['weight'], color='orange', label='Training predictions')
    #plt.plot(dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:].index, dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['weight'], color='b', label='Actual Weight')

    plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')
    plt.grid(which='major', color='#cccccc', alpha=0.5)

    plt.legend(shadow=True)
    plt.title('Predicciones y peso real', family='Arial', fontsize=12)
    plt.xlabel('Timeline', family='Arial', fontsize=10)
    plt.ylabel('Weight Value', family='Arial', fontsize=10)
    plt.show()
def plot_data(dataset_train, plot_features):
    plot_cols = ['temperature', 'humidity', 'season', 'weight'] 
    for i in plot_cols:
        plt.plot(plot_features, dataset_train[i])
        plt.title("Grafica variables ambientales")
        plt.legend(i)
        plt.xlabel('Date', family='Arial', fontsize=10)
        plt.ylabel(i, family='Arial', fontsize=10)
        plt.show()
    
def main():
    data_file = sys.argv[1]
    print("Data File--------- ", data_file)
    if data_file.endswith(('.csv')):
        X_train, y_train, sc_predict, n_future, n_past, dataset_train, datelist_train, dataset_train_timeindex, plot_features = data_processing(data_file)
        model = build_model_st(dataset_train, n_past)
    else:
        print("Enter Valid File Name")
        sys.exit()
    if sys.argv[2] == 'train':
        print("Train Model ---------")
        plot_data(dataset_train, plot_features)
        history  = model_train(model, X_train, y_train)
        plot_loss(history)
        PREDICTIONS_FUTURE, PREDICTION_TRAIN, datelist_future_ = model_prediction(datelist_train, model, n_future, n_past, X_train, sc_predict)
        PREDICTIONS_FUTURE.to_csv("Predictions_future.csv",  header=['Future Predictions'], index_label='Date')
        PREDICTION_TRAIN.to_csv("Predictions_train.csv",  header=['Train Predictions'],index_label='Date')
        plot_predictions(PREDICTIONS_FUTURE, PREDICTION_TRAIN, dataset_train_timeindex)
    elif sys.argv[2] == 'predict':
        print("Model Predictions ---------")
        plot_data(dataset_train, plot_features)
        best_model = load_best_model('best_model.h5')
        PREDICTIONS_FUTURE, PREDICTION_TRAIN, datelist_future_ = model_prediction(datelist_train, best_model, n_future, n_past, X_train, sc_predict)
        PREDICTIONS_FUTURE.to_csv("Predictions_future.csv",  header=['Future Predictions'], index_label='Date')
        PREDICTION_TRAIN.to_csv("Predictions_train.csv",  header=['Train Predictions'],index_label='Date')
        plot_predictions(PREDICTIONS_FUTURE, PREDICTION_TRAIN, dataset_train_timeindex)
    elif sys.argv[2] == 'retrain':
        print("Train Model on New Data ---------")
        plot_data(dataset_train, plot_features)
        best_model = load_best_model('best_model.h5')
        history  = model_train(best_model, X_train, y_train)
        plot_loss(history)
        PREDICTIONS_FUTURE, PREDICTION_TRAIN, datelist_future_ = model_prediction(datelist_train, best_model, n_future, n_past, X_train, sc_predict)
        PREDICTIONS_FUTURE.to_csv("Predictions_futurer.csv",  header=['Future Predictions'], index_label='Date')
        PREDICTION_TRAIN.to_csv("Predictions_trainr.csv",  header=['Train Predictions'],index_label='Date')
        plot_predictions(PREDICTIONS_FUTURE, PREDICTION_TRAIN, dataset_train_timeindex)
    else:
        print("\nPlease Enter Valid Second Argument:  train  or  predict or  retrain")



if __name__ == '__main__':
    main()


