import matplotlib.pyplot as plt
import streamlit as st

def distribution(data, column: str, num: int):
    data= data[f"{column}"]
    # Crear el gráfico de distribución
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, bins=10, edgecolor='black')

    # Resaltar la barra correspondiente al valor del slider
    for i in range(len(bins) - 1):
        if bins[i] <= num < bins[i + 1]:
            patches[i].set_facecolor('red')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)