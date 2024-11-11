import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")

class ProcesadorDeDatos:
    def __init__(self):
        self.df_final = None
        self.verificacion_exitosa = False
        self.dir = list(range(1, 361, 1))
        self.dir_graf=list(range(10, 370, 10))
        self.dir_array = np.array(self.dir)
        self.agrupar_direc = 10
        self.ordenado = None
        self.viento_calma = None
        self.suma_total_frec = None
        self.v_c_list = []
        self.lista_de_frecuencias = []
        self.max_tablas = []
        self.tipo = None
        self.agrupar_direc = 10
        self.totales=None
        self.coheficiente=None

    def cargar_archivo(self, archivo):
        """Carga un archivo y verifica su validez."""
        if archivo is None:
            st.error("No se seleccionó ningún archivo.")
            return

        try:
            self.df_final = self.read_file(archivo)
            self.verificar()  # Verificar el archivo
            
            # Si la verificación es exitosa, aplicar modificaciones
            self.df_final.columns = ['Direccion', 'Nudos']
            self.df_final['Direccion'] = self.df_final['Direccion'].replace(0, 360)
            self.verificacion_exitosa = True
            st.success("El archivo se procesó correctamente.")
        
        except ArchivoInvalidoError as e:
            st.error(f"Error en los Datos: {str(e)}")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {str(e)}")

    def read_file(self, archivo):
        """Lee el archivo y devuelve un DataFrame."""
        if archivo.name.endswith('.xlsx'):
            return pd.read_excel(archivo)
        elif archivo.name.endswith('.csv'):
            return pd.read_csv(archivo, sep=None, engine='python')
        else:
            raise ArchivoInvalidoError("El archivo debe ser un .xlsx o .csv")

    def verificar(self):
        """Verifica que el DataFrame cumpla con los requisitos."""
        if self.df_final is None or self.df_final.empty:
            raise ArchivoInvalidoError("El DataFrame está vacío.")

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
        
    def nu_tabla(self, limite, pista):
        if self.df_final is None:
            st.error("Debe cargar un archivo primero.")
            return
        elif not self.verificacion_exitosa:
            st.error("El archivo cargado no cumple con el formato deseado.")
            return
        
        df_nu_tabla=self.df_final.copy()
        total_datos=len(df_nu_tabla)
        df_nu_tabla.iloc[:, 0] = df_nu_tabla.iloc[:, 0].where(df_nu_tabla.iloc[:, 0] != 0, 360)
        df_nu_tabla.columns = ['Direccion', 'Intensidad (kt)']

        diferencia=df_nu_tabla["Direccion"] - pista
        dif_rad= np.radians(diferencia)  # Resta del índice al valor
        seno = np.sin(dif_rad) 
        seno=np.abs(seno) 
        y=seno*df_nu_tabla["Intensidad (kt)"]
        comp_transv=pd.DataFrame(y,columns=["Intensidad\nComponente transversal"])
        la_tabla = pd.concat([df_nu_tabla, comp_transv], axis=1)

        self.frec_con_limit=len(la_tabla[la_tabla["Intensidad\nComponente transversal"]<=limite])
        self.coheficiente_nu=round((self.frec_con_limit/total_datos)*100,2)

        conversion=la_tabla["Intensidad\nComponente transversal"]*1.852
        
        la_tabla["intensidad Componente Transversal (km/h)"] = conversion

        return la_tabla
    
    def nu_tabla_deca(self, limite, pista):
        if self.df_final is None:
            st.error("Debe cargar un archivo primero.")
            return
        elif not self.verificacion_exitosa:
            st.error("El archivo cargado no cumple con el formato deseado.")
            return
        
        df_nu_tabla=self.df_final.copy()
        total_datos=len(df_nu_tabla)
        df_nu_tabla.iloc[:, 0] = df_nu_tabla.iloc[:, 0].where(df_nu_tabla.iloc[:, 0] != 0, 36)
        df_nu_tabla.columns = ['Direccion', 'Intensidad (kt)']
        df_nu_tabla["Direccion"] =  df_nu_tabla["Direccion"] *10
        diferencia=df_nu_tabla["Direccion"] - pista
        dif_rad= np.radians(diferencia)  # Resta del índice al valor
        seno = np.sin(dif_rad) 
        seno=np.abs(seno) 
        y=seno*df_nu_tabla["Intensidad (kt)"]
        comp_transv=pd.DataFrame(y,columns=["Intensidad\nComponente transversal"])
        la_tabla = pd.concat([df_nu_tabla, comp_transv], axis=1)

        self.frec_con_limit=len(la_tabla[la_tabla["Intensidad\nComponente transversal"]<=limite])
        self.coheficiente_nu=round((self.frec_con_limit/total_datos)*100,2)

        conversion=la_tabla["Intensidad\nComponente transversal"]*1.852
        
        la_tabla["intensidad Componente Transversal (km/h)"] = conversion

        return la_tabla



if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'nueva_tabla_1' not in st.session_state:
    st.session_state['nueva_tabla_1'] = None

with st.container():
    uploaded_file = st.sidebar.file_uploader("Seleccionar archivo Excel o CSV", type=["xlsx", "csv"], key="file_uploader_1")

if uploaded_file is not None:
    st.session_state['data_loaded'] = True
    resultados = ProcesadorDeDatos()
    resultados.cargar_archivo(uploaded_file)
    if resultados.verificacion_exitosa:
        st.session_state['resultados'] = resultados 
        st.write("DATOS ORIGINALES")
        st.dataframe(resultados.df_final)
        
        intervalos = st.sidebar.selectbox("Seleccione intervalo (knots)", [1, 3, 5, 10],key="intervalos")
        dir_pista = st.sidebar.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
        limites = st.sidebar.number_input("Ingrese limites en (knots)", min_value=10, max_value=40, value=10, key="limites")
        nueva_tabla_1=resultados.nu_tabla_deca(limites,dir_pista)
        st.session_state['nueva_tabla_1'] = nueva_tabla_1
        
        if st.button("Resultado"):
            nueva_tabla_1 = resultados.nu_tabla_deca(limites, dir_pista)
            st.session_state['nueva_tabla_1'] = nueva_tabla_1

if st.session_state['nueva_tabla_1'] is not None:
    st.write(f"COMPONENTES TRANSVERSALES CON PISTA {dir_pista} Y LIMITE (kt) {limites}")
    st.dataframe(st.session_state['nueva_tabla_1'][["intensidad Componente Transversal (km/h)", "Intensidad\nComponente transversal"]])
else:
    st.write("No se ha generado la tabla de resultados.")



#if st.session_state['data_loaded'] and st.session_state['resultados'] and st.session_state['resultados'].verificacion_exitosa and st.session_state['nueva_tabla_1']:
df = nueva_tabla_1

if st.button("sumas"):
    
    # Crear una lista de opciones para seleccionar una fila, usando los índices del DataFrame
    filas = [f"Fila {i}" for i in df.index]
    fila_seleccionada = st.selectbox("Seleccione una fila para mostrar sus valores:", filas)
    
    # Extraer la fila seleccionada
    indice_seleccionado = int(fila_seleccionada.split()[-1])  # Extraer el índice numérico de la opción
    fila_df = df.loc[[indice_seleccionado]]  # Crear un DataFrame solo con la fila seleccionada
    
    # Crear un DataFrame solo con las columnas 'Direccion' y 'Nudos' de la fila seleccionada
    fila_df = fila_df[['Direccion', 'Nudos']]
    st.write("DataFrame con solo Dirección y Nudos de la fila seleccionada:")
    st.dataframe(fila_df)

if 'data_loaded1' not in st.session_state:
    st.session_state['data_loaded1'] = False
    st.session_state['result1'] = None

if st.button("sumas"):
    st.session_state['data_loaded1'] = True
    suma=fila_df.sum().sum()
    st.write(f"resultado: {suma}")  # Datos de ejemplo
