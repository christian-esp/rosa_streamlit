import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ArchivoInvalidoError(Exception):
    """Excepción levantada para errores de datos en el archivo."""
    pass

class ProcesadorDeDatos:
    def __init__(self):
        self.df_final = None
        self.verificacion_exitosa = False
        self.dir = list(range(1, 361, 1))
        self.dir_array = np.array(self.dir)
        self.agrupar_direc = 10
        self.ordenado = None
        self.viento_calma = None
        self.suma_total_frec = None
        self.v_c_list = []
        self.lista_de_frecuencias = []
        self.max_tablas = []

    def cargar_archivo(self, archivo):
        if archivo is not None:
            try:
                if archivo.name.endswith('.xlsx'):
                    self.df_final = pd.read_excel(archivo)
                elif archivo.name.endswith('.csv'):
                    self.df_final = pd.read_csv(archivo, sep=None, engine='python')

                verificar = self.verificar()
                if verificar == "todo ok":
                    self.verificacion_exitosa = True
                    self.df_final.columns = ['Direccion', 'Nudos']
                    self.df_final.iloc[:, 0] = self.df_final.iloc[:, 0].where(self.df_final.iloc[:, 0] != 0, 360)
                    st.success("El archivo se procesó correctamente.")
                else:
                    st.error(verificar)
            except ArchivoInvalidoError as e:
                st.error(f"Error en los Datos: {str(e)}")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {str(e)}")
        else:
            st.error("No se seleccionó ningún archivo")

    def verificar(self):
        errores = []
        if self.df_final.shape[1] < 2:
            errores.append("El archivo debe tener al menos dos columnas con títulos en la primera fila.")
        if self.df_final.columns[0] == 0 or self.df_final.columns[1] == 0:
            errores.append("La primera fila debe contener los títulos de las columnas, no datos.")
        if not pd.api.types.is_numeric_dtype(self.df_final.iloc[:, 0]):
            errores.append("La columna 'Direcciones' contiene valores no numéricos.")
        if not pd.api.types.is_numeric_dtype(self.df_final.iloc[:, 1]):
            errores.append("La columna 'Nudos' contiene valores no numéricos.")
        if self.df_final.iloc[:, 0].isnull().any():
            errores.append("La columna de Direcciones contiene valores nulos.")
        if self.df_final.iloc[:, 1].isnull().any():
            errores.append("La columna de Nudos contiene valores nulos.")

        if errores:
            raise ArchivoInvalidoError("\n".join(errores))
        
        return "todo ok"
    

    def agrupar(self, intervalo):
        if self.df_final is None:
            st.error("Debe cargar un archivo primero.")
            return
        elif not self.verificacion_exitosa:
            st.error("El archivo cargado no cumple con el formato deseado.")
            return

        conteo_viento_calma = self.df_final[self.df_final["Nudos"] == 0].shape[0]
        conteo_total = self.df_final[self.df_final["Nudos"] > 0].shape[0]

        self.viento_calma = conteo_viento_calma
        self.suma_total_frec = conteo_total

        intervalos_nudos = np.arange(0, self.df_final['Nudos'].max() + intervalo, intervalo)
        
        self.df_final['Direccion'] = self.redondear_personalizado(self.df_final['Direccion'])
        
        self.df_final.iloc[:, 0] = self.df_final.iloc[:, 0].where(self.df_final.iloc[:, 0] != 0, 360)
        self.df_final.sort_values(by='Direccion', inplace=True)
        self.df_final['Intervalo_Nudos'] = pd.cut(self.df_final['Nudos'], bins=intervalos_nudos, right=False)

        self.df_final['Intervalo_Nudos'] = self.df_final['Intervalo_Nudos'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
        self.ordenado = self.df_final.groupby(['Direccion', 'Intervalo_Nudos'], observed=False).size().unstack(fill_value=0)
        self.ordenado = self.ordenado.applymap(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)

        return self.ordenado

    def redondear_personalizado(self, numeros):
        parte_decimal = numeros - np.floor(numeros)
        return np.where(parte_decimal >= 0.5, np.ceil(numeros), np.floor(numeros))

class App:
    def __init__(self):
        self.resultados=ProcesadorDeDatos()
        self.valor_seleccionado = None
        self.intervalo=None
        


    def mostrar_widgets(self):
        uploaded_file = st.file_uploader("Seleccionar archivo Excel o CSV", type=["xlsx", "csv"], key="file_uploader_1")
        if uploaded_file is not None:
            self.resultados.cargar_archivo(uploaded_file)
        valores = [1,3,5,10]
        self.valor_seleccionado = st.selectbox("Seleccione un valor", valores)
        st.write(f"Valor seleccionado: {self.valor_seleccionado}")
     
# Intervalo de agrupación
        self.intervalo = st.number_input("Ingrese el intervalo para agrupar (en Nudos)", min_value=1, value=10)
        if st.button("Agrupar"):
            resultados = self.resultados.agrupar(self.intervalo)
            if resultados is not None:
                st.write("Resultados de Agrupación:")
                st.dataframe(resultados)


def main():
    st.title("ROSA DE LOS VIENTOS")

    # Crear una instancia de la clase InterfazStreamlit
    interfaz = App()

    # Llamar al método que muestra los widgets de la interfaz
    interfaz.mostrar_widgets()

if __name__ == "__main__":
    main()