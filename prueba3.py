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
#st.sidebar.image(image_path, width=200) 
st.sidebar.markdown(
    f"""
    <style>
        .sidebar-img {{
            display: flex;
            justify-content: center;
            margin-top: -90px;  /* Ajusta este valor para mover más hacia arriba */
            margin-bottom: -2px; /* Espaciado con otros elementos */
        }}
        .sidebar-img img {{
            width: 150px;  /* Ancho de la imagen */
            height: auto;
        }}
    </style>
    <div class="sidebar-img">
        <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        .main-title {
            margin-top: -55px;  /* Ajusta este valor para mover más hacia arriba */
            text-align: center; /* Centra el título horizontalmente */
        }
    </style>
    <h1 class="main-title">ANÁLISIS DE VIENTOS</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Cambiar el ancho de la barra lateral */
    [data-testid="stSidebar"] {
        width: 250px;  /* Fija el ancho deseado */
        min-width: 250px;  /* Establece un mínimo consistente */
        max-width: 250px;  /* Evita que se pueda redimensionar */
    }
    
    /* Ajustar widgets dentro de la barra lateral */
    .stSidebar div.stSlider, .stSidebar div.stNumberInput, .stSidebar div.stFileUploader {
        width: 100% !important;  /* Ajustar el ancho de los widgets */
        margin: 0 auto;  /* Centrar los widgets */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stExpander"] > div:nth-child(2) {
        overflow-y: scroll;
        max-height: 500px; /* Cambia el tamaño máximo */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        /* Mover el file_uploader hacia arriba */
        .stFileUploader {
            margin-top: -80px;  /* Ajusta el valor según necesites */
        }
    </style>
    """,
    unsafe_allow_html=True
)
 
class ProcesadorDeDatos:
    def __init__(self):
        self.df_final = None
        self.verificacion_exitosa = False
        self.data_procesada=False
        self.df_nu_tabla=None
    
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
        
    
    def nu_tabla_deca(self, pista,data_base):
        if data_base is None:
            st.error("Debe cargar un archivo primero.")
            return
        elif not self.verificacion_exitosa:
            st.error("El archivo cargado no cumple con el formato deseado.")
            return
        
        diferencia=data_base["Direccion"] - pista
        dif_rad= np.radians(diferencia)  # Resta del índice al valor
        seno = np.sin(dif_rad) 
        seno=np.abs(seno) 
        data_base["Intensidad (kt)"] = pd.to_numeric(data_base["Intensidad (kt)"], errors='coerce')
        y=seno*data_base["Intensidad (kt)"]
        y.name = "Componente Transversal (kt)"

        return y

    def redondear_personalizado(self, numeros):
        parte_decimal = numeros - np.floor(numeros)
        return np.where(parte_decimal >= 0.5, np.ceil(numeros), np.floor(numeros))
    
    def coheficiente(self,limite,comp_transv):
        if comp_transv is None:
            st.error("Los datos no han sido procesados correctamente. Por favor, verifique los pasos anteriores.")
            return None
        datos_totales= len(comp_transv)
        datos_filtrados=(comp_transv<=limite).sum()
        datos_filtrados1=(datos_filtrados / datos_totales)*100
        #len_datos_filtrados=len(datos_filtrados)
        #coheficiente= (len_datos_filtrados/datos_totales) * 100
        return datos_filtrados1.round(2)

    def tabla_grafico(self,tabla):
        tabla_para_grafico=tabla
        total_datos_orgin=len(tabla_para_grafico)
        
        tabla_para_grafico['Intensidad (kt)'] = pd.to_numeric(tabla_para_grafico['Intensidad (kt)'], errors='coerce')

        intervalos_nudos = [-0.1, 10] + list(np.arange(20, tabla_para_grafico['Intensidad (kt)'].max() + 10, 10))
        tabla_para_grafico['Direccion'] = self.redondear_personalizado(tabla_para_grafico['Direccion'])
        
        #df_agrupar1.iloc[:, 0] = df_agrupar1.iloc[:, 0].where(df_agrupar1.iloc[:, 0] != 0, 36)
        tabla_para_grafico.sort_values(by='Direccion', inplace=True)
        tabla_para_grafico['Intensidad (kt)'] = pd.cut(tabla_para_grafico['Intensidad (kt)'], bins=intervalos_nudos, right=True)

        tabla_para_grafico['Intensidad (kt)'] = tabla_para_grafico['Intensidad (kt)'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
        tabla_para_grafico['Direccion'] = ((tabla_para_grafico['Direccion'] - 1) // 10 * 10 + 10).astype(int)

        tabla_para_grafico = tabla_para_grafico.groupby(['Direccion', 'Intensidad (kt)']).size().unstack(fill_value=0)

        for col in tabla_para_grafico.columns:
            tabla_para_grafico[col] = pd.to_numeric(tabla_para_grafico[col], errors='coerce').fillna(0)

    # Convertir a float para operaciones de porcentaje
        #tabla_para_grafico = tabla_para_grafico.astype(float)
        #total_datos_orgin
        #suma_con_ceros=len(tabla_para_grafico)

    # Realizar el cálculo de porcentaje
        tabla_para_grafico = (tabla_para_grafico / total_datos_orgin) * 100

    # Redondear a 2 decimales
        #tabla_para_grafico = tabla_para_grafico.round(2)

        return tabla_para_grafico.round(2)

    
    def grafico(self,pista,limite,df_graf):
        fig = go.Figure()
        #df_graf=self.df_nu_tabla
       
        # Agregar una línea que cruza desde la dirección de la pista hasta la dirección opuesta
        angulo_pista = pista    # Dirección ingresada por el usuario
        angulo_opuesto = (angulo_pista + 180) % 360  # Dirección opuesta

        limite_knots = {
            (10, 11): 107,
            (12, 13): 125,
            (14, 20): 210
        }

        # Línea que va de la dirección de la pista a su dirección opuesta
        for (lower_limit, upper_limit), dist in limite_knots.items():
            if lower_limit <= limite <= upper_limit:
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
            if lower_limit <= limite <= upper_limit:
   
                for offset in [-10, 10]:  # Desplazamientos para las líneas
                    fig.add_trace(go.Scatterpolar(
                        r=[r, r],  # Desde el centro hasta el borde
                        theta=[angulo_pista + offset, angulo_opuesto - offset],  # Ángulos de inicio y fin
                        mode='lines',
                        line=dict(color="green", width=2),
                        showlegend=False
                        
                    ))
                    

        suma_0_10 = df_graf.iloc[:,0].sum()
        if suma_0_10 > 0:  # Solo mostrar la suma si es mayor que cero
            fig.add_trace(go.Scatterpolar(
                r=[0],  # Ubicar en el centro
                theta=[0],  # Posición horizontal para el texto
                mode='text',  # Modo de texto
                text=[f"{suma_0_10:.2f}"],  # Anotación con la suma
                textposition='middle center',  # Centrar el texto en el círculo
                textfont=dict(size=20, color='black', family="Arial Black"),  # Estilo del texto
                showlegend=False  # No mostrar en la leyenda
            ))


        for i in range(1, df_graf.shape[1]):  # Empezar en 1 para ignorar la primera columna (0-10)
            intervalo_nudos = df_graf.columns[i]  # Nombre de la columna actual
            intensidades = df_graf.iloc[:, i].values  # Valores radiales (porcentajes)
            direcciones = df_graf.index  # Índices del DataFrame son las direcciones en grados

        # Calcular el radio correspondiente para el intervalo de nudos
            radio = (i) * 10  # Cada intervalo de nudos se asocia a un radio específico

        # Mostrar los valores individuales en cada dirección para los intervalos a partir de 10-20
            for j, valor in enumerate(intensidades):
                if valor >= 0.1:  # Solo mostrar si el valor es mayor que cero
                    fig.add_trace(go.Scatterpolar(
                        r=[radio+5],  # Usar el radio calculado
                        theta=[direcciones[j]],  # Dirección correspondiente
                        mode='text',  # Modo de texto
                        text=[f"{valor:.2f}"],  # Anotación con el valor individual
                        textposition='middle center',  # Centrar el texto en la dirección
                        textfont=dict(size=12, color='blue',family="Arial Black"),  # Estilo del texto
                        showlegend=False  # No mostrar en la leyenda
                    ))
                elif 0<valor<0.1:
                    fig.add_trace(go.Scatterpolar(
                        r=[radio+5],  # Usar el radio calculado
                        theta=[direcciones[j]],  # Dirección correspondiente
                        mode='text',  # Modo de texto
                        text=["+"],  # Anotación con el valor individual
                        textposition='middle center',  # Centrar el texto en la dirección
                        textfont=dict(size=13, color='black', family='Arial Black'),  # Estilo del texto
                        showlegend=False  # No mostrar en la leyenda
                    ))

        
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
                    tickvals=[10, 20, 30, 40, 50],
                )
            ),
            
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

    def grafico1(self,tablita_df,pista,limite):
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
        #limites=limite
        # Línea que va de la dirección de la pista a su dirección opuesta
        for (lower_limit, upper_limit), dist in limite_knots.items():
            if lower_limit <= limite <= upper_limit:
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
            if lower_limit <= limite <= upper_limit:
   
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
        tablita_df = tablita_df.to_frame().T 
        for _, row in tablita_df.iterrows():
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

uploaded_file = st.file_uploader("", type=["xlsx", "csv"], key="file_uploader_1")
dir_pista = st.sidebar.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
limites = st.sidebar.number_input("Ingrese limites en (knots)", min_value=10, max_value=40, value=10, key="limites")


if "tabla_original" not in st.session_state:
    st.session_state["tabla_original"] = None
if "tabla_procesada" not in st.session_state:
    st.session_state["tabla_procesada"] = None
if "fila_seleccionada" not in st.session_state:
    st.session_state["fila_seleccionada"] = None

#SI APRETO EL BOTON DE CARGA DE ARCHIVO:
resultados = ProcesadorDeDatos()
header_container = st.container()
# Contenedor para los resultados
results_container = st.container()

with header_container:
    #st.title("Mi Aplicación")
    #uploaded_file = st.file_uploader("", type=["xlsx", "csv"], key="file_uploader_1")
    #tabla_original=resultados.cargar_archivo(uploaded_file)
    #copia=tabla_original.copy()
    page = st.selectbox("Selecciona una opción", ["Ver Resultados", "Pruebas individuales"])


if uploaded_file is not None:
    tabla_original=resultados.cargar_archivo(uploaded_file)
    copia=tabla_original.copy()
    if page=="Ver Resultados":
        with results_container:
            st.subheader("Resultados")
            col1, col2 = st.columns(2)

            with col1:
                st.expander("Tabla Original").dataframe(tabla_original)

            with col2:
                pa_el_box=resultados.nu_tabla_deca(dir_pista,copia)

                st.expander("Tabla Procesada").dataframe(pa_el_box)
            
            cohe=resultados.coheficiente(limites,pa_el_box)
            if cohe is not None:
                st.markdown(f"**Coeficiente de utilización:** {cohe}%")
       
            df_graf=resultados.tabla_grafico(copia)
            #st.dataframe(df_graf)
            with st.expander("GRAFICO"):
                resultados.grafico(dir_pista,limites,df_graf)

    if page=="Pruebas individuales": 
        with results_container:          
    
            st.subheader("Pruebas individuales")
            col1, col2 = st.columns(2)

            with col1:
                st.expander("Tabla Original").dataframe(tabla_original)

            with col2:
                pa_el_box=resultados.nu_tabla_deca(dir_pista,copia)

                
            fila_seleccionada = st.selectbox(
                "Seleccione una fila para mostrar sus valores:",
                options=copia.index,
                index=(0),  # Default a la fila seleccionada
                key="fila_seleccionada")
    

            if fila_seleccionada:
                componente = copia.iloc[fila_seleccionada] 
                #componente = componente.to_frame()
 
                st.dataframe(componente)
                diferencia=componente["Direccion"] - dir_pista
                dif_rad= np.radians(diferencia)  # Resta del índice al valor
                seno = np.sin(dif_rad) 
                seno=np.abs(seno) 
                #data_base["Intensidad (kt)"] = pd.to_numeric(data_base["Intensidad (kt)"], errors='coerce')
                y=seno*componente["Intensidad (kt)"]
                st.write()
                st.write(f"Componente Transversal (kt): {y.round(3)}")
                
                if y <= limites:
                    st.write("Coheficiente de utilización: 100%")
                else:
                    st.write("Coheficiente de utilización: 0")
                with st.expander("GRAFICO_1"):
                    resultados.grafico1(componente,dir_pista,limites)