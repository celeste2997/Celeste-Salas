import sys
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import tensorflow as tf
import datetime as dt
from datetime import datetime
from keras.models import load_model
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
import numpy as np

from data_preparation import data_processing
from build_model import build_model_st, plot_predictions, model_prediction, load_best_model
best_model = load_best_model('best_model.h5')

global data_file
global pred_h
def main():

	st.title("Predicción del Peso de la Colmena Artificial de abejas Apis Melífera.")
	menu =["Inicio"]
	choice =st.sidebar.selectbox("Predicción",menu)


	if choice == "Inicio":
		st.subheader("A continuación se realizará la predicción del peso implementando un modelo Bidireccional LSTM entrenado y guardado en la nube con el fin de reentrenarlo a través de interfaz gráfica usando nuevos datos de la colmena.") 
		st.subheader("Por favor escoja un archivo *.csv que contenga el registro de las variables ambientales tomadas de la colmena artificial.")

		data_file = st.file_uploader("ARCHIVO.csv", type=['csv'])
		if data_file is not None:
			file_details = {"filename": data_file.name}

			st.write("La actualización de la predicción se realiza cada vez se ingrese un archivo y cuando hay actualización en los días pasados y futuros ingresados posteriormente.")

			dataset_train1, datelist_train, dataset_train_timeindex, df1,training_set= data_processing(data_file)
			
			dias_reg =len(datelist_train)
			d_r=int((dias_reg)/24)
			
			val_max = int((d_r/2)-1)

			st.subheader("Tabla base de datos cargada por el usuario:")
			st.dataframe(df1)
			sc = StandardScaler()
			training_set_scaled = sc.fit_transform(training_set)

			sc_predict = StandardScaler()
			sc_predict.fit_transform(training_set[:, 0:1])
					# Creating a data structure with 72 timestamps and 1 output
			X_train = []
			y_train = []   

			st.subheader("La cantidad del número de días pasados y futuros aceptados para el entrenamiento dependen de la cantidad de datos registrados en la base de datos subida por el usuario.")
			st.subheader('Número de días registrados en el archivo es: ')
			st.write(d_r)
			st.subheader("Seleccione el número de días pasados a tener en cuenta para la predicción a futuro:")

			number1 = (st.number_input("# días",max_value=val_max, min_value=3, step=1))
				#st.subheader('Número de días a predecir: ',number)
				
			past =(int(number1)*24)
			st.subheader("Seleccione el número de días para predecir el peso de la colmena en el futuro:")

			number = (st.number_input(" # días",max_value=30, min_value=1, step=1))
				#st.subheader('Número de días a predecir: ',number)
				
			pred_h =(int(number)*24)



			if pred_h is not None:
				#global n_future   # Number of days we want top predict into the future
				n_past = past     # Number of past days we want to use to predict the future
					#input_future =0
				n_future = pred_h
					#print(input_future, n_future)
				for i in range(n_past, len(training_set_scaled) - n_future +1):
					X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train1.shape[1]])
					y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
				X_train, y_train = np.array(X_train), np.array(y_train)

				print('X_train shape == {}.'.format(X_train.shape))
				print('y_train shape data pre == {}.'.format(y_train.shape))


				START_DATE_FOR_PLOTTING = '2017-01-07'

				 #Predicciones

				PREDICTIONS_FUTURE, PREDICTION_TRAIN, datelist_future_ = model_prediction(datelist_train, best_model, n_future, n_past, X_train, sc_predict)
				
				#dfp=pd.DataFrame(PREDICTIONS_FUTURE)
				#st.download_button(label='Descargar predicción',data=PREDICTIONS_FUTURE,file_name='Predicción_GUI.txt')

				st.write("Peso predicho en Kg",PREDICTIONS_FUTURE)

				fig1 = plt.figure(figsize=(5,3)) # try different values
				ax = plt.axes()
				#ax.legend()
				st.subheader("Peso predicho por las redes Bidireccionales LSTM")
				ax.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['weight'], color='r')

				plt.gcf().autofmt_xdate()
				plt.xlabel('Tiempo')
				plt.ylabel('Peso [Kg]')
				st.pyplot(fig1)

				
				st.subheader("Gráfica predicción del peso con los datos cargados en el archivo .csv")

				ts = dataset_train1["weight"] 
				#dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['weight']
				fig1 = plt.figure(figsize=(5,3)) # try different values
				ax = plt.axes()
				#ax.legend()
				ax.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['weight'], color='r')
				ax.plot(dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:].index, dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['weight'])
				ax.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')
				plt.gcf().autofmt_xdate()
				plt.xlabel('Tiempo')
				plt.ylabel('Peso [Kg]')
				st.pyplot(fig1)
				

				st.subheader("Gráfica variables ambientales")
				st.subheader("A continuación se grafican las variables ambientales de la colmena artficial")

				
				# G R Á F  I C A   E S T A C I Ó N 



				st.subheader("Estación del año")
				st.write("Invierno(0) - Primavera (0.5) - Verano (1) - Otoño (0.25)")
				ts = dataset_train1["season"]
				fig1 = plt.figure(figsize=(4,3)) # try different values
				ax = plt.axes()
				            #ax.legend()
				ax.plot(datelist_train, ts)
				plt.gcf().autofmt_xdate()
				plt.xlabel('Tiempo')
				st.pyplot(fig1)
				
				# G R Á F I C A    P E S O 

				st.subheader("Peso Colmena")
				ts = dataset_train1["weight"]
				fig1 = plt.figure(figsize=(5,3)) # try different values
				ax = plt.axes()
				            #ax.legend()
				ax.plot(datelist_train, ts)
				plt.gcf().autofmt_xdate()
				plt.xlabel('Tiempo')
				plt.ylabel('Peso [Kg]')
				st.pyplot(fig1)

                # G R Á F I C A   T E M P E R A T U R A


				st.subheader("Temperatura Colmena")
				ts = dataset_train1["temperature"]
				fig1 = plt.figure(figsize=(4,3)) # try different values
				ax = plt.axes()
				#ax.legend()
				ax.plot(datelist_train, ts)
				plt.gcf().autofmt_xdate()
				plt.xlabel('Tiempo')
				plt.ylabel('Temperatura [°C]')
				st.pyplot(fig1)

				# G R Á F I C A   H U M E D A D


				st.subheader("Humedad Colmena")
				ts = dataset_train1["humidity"]
				fig1 = plt.figure(figsize=(4,3)) # try different values
				ax = plt.axes()
				#ax.legend()
				ax.plot(datelist_train, ts)
				plt.gcf().autofmt_xdate()
				plt.xlabel('Tiempo')
				plt.ylabel('Humedad [%]')
				st.pyplot(fig1)

		

if __name__=='__main__':
	main()
