import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import distribution
data= pd.read_csv("data_etiquetada.csv")

st.write(f'<p class="big-font"> Simulación de estimulación ovárica</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])


with col1:
    
    variable="edad paciente"
    age = st.slider(f"Ingrese la edad  del paciente:                                  \n.\n ", 18, 50) 
    distribution(data,variable,age) 
    st.write(f"El paciente tiene: {age} años")

with col2:
    variable= "amh"
    amh_log = col2.slider("Ingrese cantidades obtenidas de \n la hormona antimuleriana en sangre:", 0.01, 13.0) 
    distribution(data,variable,amh_log) 
    col2.write(f"Antimulleriana en sangre: {amh_log} ng/ml")
with col3:
    variable= "total rfa"
    total_rfa = col3.slider(f"Ingrese el to tal de recuentos de foliculos antrales", 1, 20) 
    distribution(data,variable,total_rfa) 
    col3.write(f"La cantidad de foliculos antrales es: {total_rfa}")


df_pred = pd.DataFrame([[age,amh_log,total_rfa]])#

columns= ['edad paciente','amh_log','total rfa']

df_pred.columns = columns

#Transformaciones de los datos
#Transformación Logarítmica
def transform(data):
    return np.log(data)

df_pred['edad paciente'] = df_pred['edad paciente']

df_pred['amh_log'] = df_pred['amh_log'].apply(transform)

df_pred['total rfa'] = df_pred['total rfa']

#Carga de modelos
model1 = joblib.load('RL1_model.pkl')
model2 = joblib.load('RL2_model.pkl')

#Ensamble de modelos 
prediction1 = model1.predict(df_pred)
prediction2 = model2.predict(df_pred)
# Ejemplo de función para obtener predicciones de los modelos
def obtener_predicciones(df_pred):
    prediction1 = model1.predict(df_pred)
    prediction2 = model2.predict(df_pred)
    
    if prediction1 == 0:
        result = 0
    elif prediction1 == 1:
        if prediction2 == 1:
            result = 2
        elif prediction2 == 0:
            result = 1
    else:
        result = None  # Manejar casos inesperados
    
    return prediction1, prediction2, result

# Interfaz de la aplicación con Streamlit
def main():
    st.title('Aplicación de Predicción')
    
    # Aquí deberías tener el código para cargar y preparar tus datos de entrada (df_pred)
    # df_pred = cargar_datos_para_prediccion()  # Ejemplo
    
    #if st.button('Predict'):
    prediction1, prediction2, result = obtener_predicciones(df_pred)
    
    # Convertir el resultado a texto para mostrar en la interfaz
    if result == 0:
        result_str = "<=4"
    elif result == 1:
        result_str = "5-9"
    elif result == 2:
        result_str = ">9"
    else:
        result_str = "No disponible"
    
    # Mostrar los resultados en la interfaz
    st.write(f'<p class="big-font">Predicción para el modelo 1: {prediction1}, para el modelo 2: {prediction2}, para el ensamble del modelo: {result_str}</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
