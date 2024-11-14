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
    

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'nueva_tabla_1' not in st.session_state:
    st.session_state['nueva_tabla_1'] = None
if 'tabla_original' not in st.session_state:
    st.session_state['tabla_original'] = None
if 'fila_seleccionada' not in st.session_state:
    st.session_state['fila_seleccionada'] = None

with st.container():
    uploaded_file = st.sidebar.file_uploader("Seleccionar archivo Excel o CSV", type=["xlsx", "csv"], key="file_uploader_1")

if uploaded_file is not None:
    st.session_state['data_loaded'] = True
    resultados = ProcesadorDeDatos()
    resultados.cargar_archivo(uploaded_file)
    api=App()


    if resultados.verificacion_exitosa:
        st.session_state['resultados'] = resultados 
        st.write("DATOS ORIGINALES")
        if (resultados.df_final['Direccion'].abs() >= 100).any():
            st.dataframe(resultados.df_final)
        else:
            resultados.df_final['Direccion']=resultados.df_final['Direccion']*10
            st.dataframe(resultados.df_final)
        
        intervalos = st.sidebar.selectbox("Seleccione intervalo (knots)", [1, 3, 5, 10],key="intervalos")
        dir_pista = st.sidebar.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
        limites = st.sidebar.number_input("Ingrese limites en (knots)", min_value=10, max_value=40, value=10, key="limites")
        #nueva_tabla_1=resultados.nu_tabla_deca(limites,dir_pista)
        tabla_original=resultados.df_final
        
        
        st.session_state['tabla_original'] = tabla_original
        if 'nueva_tabla_1' not in st.session_state:
            st.session_state['nueva_tabla_1'] = None

        if st.button("Resultado"):
            nueva_tabla_1 = resultados.nu_tabla_deca(limites, dir_pista)
            st.session_state['nueva_tabla_1'] = nueva_tabla_1
            filas = [f"Fila {i}" for i in nueva_tabla_1.index]  # Guardar las filas en el estado
            st.session_state["fila_seleccionada"]= st.selectbox(
                    "Seleccione una fila para mostrar sus valores:",
                    st.session_state['filas'],
                    key="fila_seleccionada"
                )
# Mostrar la tabla si ya está calculada
        if 'nueva_tabla_1' in st.session_state and st.session_state['nueva_tabla_1'] is not None:
            nueva_tabla_1 = st.session_state['nueva_tabla_1']
            st.write(f"COMPONENTES TRANSVERSALES CON PISTA {dir_pista} Y LIMITE (kt) {limites}")
            nueva_tabla_1["Direccion"]=nueva_tabla_1["Direccion"]/10
            st.dataframe(nueva_tabla_1)
           

    # Mostrar el selectbox
            if 'filas' in st.session_state:
                fila_seleccionada = st.selectbox(
                    "Seleccione una fila para mostrar sus valores:",
                    st.session_state['filas'],
                    key="fila_seleccionada"
                )

                if fila_seleccionada:
            # Calcular resultados basados en la fila seleccionada
                    indice_seleccionado = int(fila_seleccionada.split()[-1])
                    fila_df = nueva_tabla_1.loc[[indice_seleccionado]]

            # Mostrar la fila seleccionada
                    st.write("DataFrame con solo Dirección y Nudos de la fila seleccionada:")
                    st.dataframe(fila_df)
                    with st.expander("grafico"):
                        #tablin=fila_df.iloc[:, 0:2]
                        api.grafico(fila_df,dir_pista, limites)

            # Calcular el resultado
                    dif = fila_df.iloc[0, 0] - dir_pista
                    diferencia_rad = np.radians(dif)
                    resultado = np.sin(diferencia_rad) * fila_df.iloc[0, 1]
                    resultado = np.abs(resultado).round(2)

            # Mostrar resultado
                    if resultado <= limites:
                        st.write(f"Resultado: {resultado}")
                    else:
                        st.write("Resultado: 0")