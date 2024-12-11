import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from math import pi
from windrose import WindroseAxes

tabla=pd.read_excel("C:/Users/cespi/Desktop/Proyecto_rosa/rosa_streamlit/rio_grande_2colum.xlsx")

tabla.columns = ['Direccion', 'Intensidad (kt)']

def redondear_personalizado(self, numeros):
    parte_decimal = numeros - np.floor(numeros)
    return np.where(parte_decimal >= 0.5, np.ceil(numeros), np.floor(numeros))
frecuencias = [10, 20, 30, 40,50]
tabla["Direccion"]= tabla["Direccion"].replace(0, 36)
tabla["Direccion"]=tabla["Direccion"]*10
direc_deca_completas = pd.DataFrame({'Direccion': list(range(10, 361, 10))})
tabla= pd.merge(direc_deca_completas,tabla,on="Direccion",how="left")
tabla["Intensidad (kt)"]=tabla["Intensidad (kt)"].fillna(0)
ax = WindroseAxes.from_ax()
ax.contourf(tabla["Direccion"], tabla["Intensidad (kt)"], bins = np.arange(0, 51, 10), cmap=cm.hot)
ax.set_yticklabels([10, 20, 30, 40, 50])  # Etiquetas para mostrar

ax.set_legend()
plt.show()
print(tabla)
