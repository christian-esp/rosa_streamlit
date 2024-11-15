import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import base64

st.set_page_config(layout="wide",page_title="Análisis de vientos")

image_path = "images/logo.png" 
 # Reemplaza con el nombre de tu imagen
st.sidebar.image(image_path, width=200) 
st.title("ANALISIS DE VIENTOS")
 
class ProcesadorDeDatos:
    def __init__(self):
        self.df_final = None
        self.verificacion_exitosa = False
        self.data_procesada=False

    
    def cargar_archivo(self, archivo):
        """Carga un archivo y verifica su validez."""
        if archivo is None:
            st.error("No se seleccionó ningún archivo.")
            return

        try:
            self.df_final = self.read_file(archivo)
            self.verificar()  # Verificar el archivo
            if not self.data_procesada:
            
            # Si la verificación es exitosa, aplicar modificaciones
                self.df_final.columns = ['Direccion', 'Intensidad (kt)']
                self.data_procesada = True
                self.verificacion_exitosa = True
                #st.success("El archivo se procesó correctamente.")
                if (self.df_final['Direccion'].abs() >= 100).any():
                    return self.df_final
                else:
                
                    self.df_final["Direccion"]= self.df_final["Direccion"].replace(0, 36)
                    self.df_final["Direccion"]=self.df_final["Direccion"]*10
                    direc_deca_completas = pd.DataFrame({'Direccion': list(range(10, 361, 10))})
                    self.df_final= pd.merge(direc_deca_completas,self.df_final,on="Direccion",how="left")
                    self.df_final["Intensidad (kt)"]=self.df_final["Intensidad (kt)"].fillna(0)
                    return self.df_final
        
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
        
    
    def nu_tabla_deca(self, limite, pista):
        if self.df_final is None:
            st.error("Debe cargar un archivo primero.")
            return
        elif not self.verificacion_exitosa:
            st.error("El archivo cargado no cumple con el formato deseado.")
            return
        
        df_nu_tabla=self.df_final.copy()
        diferencia=df_nu_tabla["Direccion"] - pista
        dif_rad= np.radians(diferencia)  # Resta del índice al valor
        seno = np.sin(dif_rad) 
        seno=np.abs(seno) 
        y=seno*df_nu_tabla["Intensidad (kt)"]
        #comp_transv=pd.DataFrame(y,columns=["Intensidad\nComponente transversal"])
        #la_tabla = pd.concat([df_nu_tabla, comp_transv], axis=1)

        #self.frec_con_limit=len(la_tabla[la_tabla["Intensidad\nComponente transversal"]<=limite])
        #self.coheficiente_nu=round((self.frec_con_limit/total_datos)*100,2)

        #conversion=la_tabla["Intensidad\nComponente transversal"]*1.852
        
        #la_tabla["intensidad Componente Transversal (km/h)"] = conversion

        return y
    
class App:
    def __init__(self):
        self.resultados=ProcesadorDeDatos()

    def grafico(self,tablita,pista,limite):
        fig = go.Figure()
        #tabla_para_puntos=tablita
        #df_graf=self.resultados.tabla_grafico()

        # Agregar una línea que cruza desde la dirección de la pista hasta la dirección opuesta
        angulo_pista = pista  # Dirección ingresada por el usuario
        angulo_opuesto = (angulo_pista + 180) % 360  # Dirección opuesta

        limite_knots = {
            (10, 11): 107,
            (12, 13): 125,
            (14, 20): 210
        }
        limites=limite
        # Línea que va de la dirección de la pista a su dirección opuesta
        for (lower_limit, upper_limit), dist in limite_knots.items():
            if lower_limit <= limites <= upper_limit:
                fig.add_trace(go.Scatterpolar(
                    r=[50, 50],  # Se extiende desde el borde interior (10) hasta el borde exterior (50)
                    theta=[angulo_pista, angulo_opuesto],  # Dirección de la pista hasta la opuesta
                    mode='lines',
                    line=dict(color="green", width=dist),
                    opacity=0.3,
                    name="Pista"
                ))

        limite_rango = {
            (10, 11): 59,
            (12, 13): 69,
            (14, 20): 116
        }
# Iterar sobre los rangos y agregar las líneas
        for (lower_limit, upper_limit), r in limite_rango.items():
            if lower_limit <= limites <= upper_limit:
   
                for offset in [-10, 10]:  # Desplazamientos para las líneas
                    fig.add_trace(go.Scatterpolar(
                        r=[r, r],  # Desde el centro hasta el borde
                        theta=[angulo_pista + offset, angulo_opuesto - offset],  # Ángulos de inicio y fin
                        mode='lines',
                        line=dict(color="green", width=2),
                        showlegend=False
                        
                    ))                   
        #x=tabla_para_puntos[["Direccion","Intensidad (kt)"]]
        tickvals = [10, 20, 30, 40, 50]
        for _, row in tablita.iterrows():
            angulo_viento = row["Direccion"]
            intensidad = row["Intensidad (kt)"]
    
    # Determinar el rango correspondiente para la intensidad
            for i in range(len(tickvals) - 1):
                limite_inferior = tickvals[i]
                limite_superior = tickvals[i + 1]
        
        # Verificar si la intensidad cae dentro del rango actual
                if limite_inferior <= intensidad <= limite_superior:
            # Calcular la posición relativa
                    posicion_relativa = (intensidad - limite_inferior) / (limite_superior - limite_inferior)
            
            # Calcular el radio correspondiente dentro del rango
                    radio = limite_inferior + (posicion_relativa * (limite_superior - limite_inferior))
            
            # Agregar el punto al gráfico
                    fig.add_trace(go.Scatterpolar(
                        r=[radio],
                        theta=[angulo_viento],
                        mode='markers',
                        marker=dict(symbol='circle', color='red', size=4),
                        showlegend=False
                    ))
                    break 

        
        # Configuración del gráfico polar
        fig.update_layout(
            polar=dict(
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    dtick=10,
                    tickmode='array',
                    tickvals=[i for i in range(0, 360, 10)],
                    ticktext=[f"{i}" if i > 0 else "360" for i in range(0, 360, 10)]
                ),
                radialaxis=dict(
                    title="Nudos",
                    range=[0, 50],  # Ajusta el rango según tus datos
                    tickvals=tickvals,
                )
            ),
            title="GRAFICO",
            width=700,
            height=700
        )
        config = {
            'scrollZoom': False,      # Desactiva el zoom con scroll
            'displayModeBar': False,  # Oculta la barra de herramientas
            'staticPlot': True        # Convierte el gráfico en estático
        }
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, config=config)
    

        
#intervalos = st.sidebar.selectbox("Seleccione intervalo (knots)", [1, 3, 5, 10],key="intervalos")
uploaded_file = st.sidebar.file_uploader("Seleccionar archivo Excel o CSV", type=["xlsx", "csv"], key="file_uploader_1")
dir_pista = st.sidebar.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
limites = st.sidebar.number_input("Ingrese limites en (knots)", min_value=10, max_value=40, value=10, key="limites")

#SI APRETO EL BOTON DE CARGA DE ARCHIVO:
col1, col2 = st.columns(2)
if uploaded_file is not None:
    resultados = ProcesadorDeDatos()
    tabla_original=resultados.cargar_archivo(uploaded_file)
    st.sidebar.success("El archivo se procesó correctamente.")
    with col1:
        with st.expander("TABLA ORIGINAL"):
            st.dataframe(tabla_original)
    

    if st.sidebar.button("Resultados"):
        if tabla_original is not None:
            tabla_procesada=resultados.nu_tabla_deca(limites, dir_pista)
            tabla_procesada=pd.DataFrame(tabla_procesada)
            tabla_procesada = tabla_procesada.rename(columns={
                tabla_procesada.columns[0]: 'Componente Transversal (kt)'
            })
            with col2:
                with st.expander("TABLA PROCESADA"):
                    st.dataframe(tabla_procesada)






        
       