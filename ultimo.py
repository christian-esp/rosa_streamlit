import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.lines import Line2D 
from st_aggrid import AgGrid, GridOptionsBuilder


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
 
class ArchivoInvalidoError(Exception):
    """Excepción personalizada para archivos inválidos."""
    pass

class ProcesadorDeDatos:
    def __init__(self):
        self.df_final = None
        self.verificacion_exitosa = False
        self.data_procesada=False
        self.df_nu_tabla=None
        self.valores_validos = set(range(10, 361, 10))
    
    
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
                direcciones_validas = set(range(0, 361, 10))  # Conjunto con las direcciones válidas (10, 20, ..., 360)
                direcciones_en_df = set(self.df_final['Direccion'])  # Conjunto con las direcciones en el DataFrame

# Paso 2: Verificar si hay direcciones que faltan
                direcciones_faltantes = direcciones_validas - direcciones_en_df
                
                
                if (self.df_final['Direccion'].abs() >= 100).any() and not direcciones_faltantes:
                    #self.df_final.loc[(self.df_final['Direccion'] >= 0) & (self.df_final['Direccion'] < 0.5), 'Direccion'] = 360
                    self.df_final["Direccion"]= self.df_final["Direccion"].replace(0, 360)
                    
                    return self.df_final
                
                elif (self.df_final['Direccion'].abs() >= 100).any() and direcciones_faltantes:
                    df_faltantes = pd.DataFrame({'Direccion': list(direcciones_faltantes), 'Intensidad (kt)': [0] * len(direcciones_faltantes)})

    # Paso 4: Concatenar las direcciones faltantes al DataFrame original
                    self.df_final = pd.concat([self.df_final, df_faltantes], ignore_index=True)
                    self.df_final["Direccion"]= self.df_final["Direccion"].replace(0, 360)

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
        y=y.to_frame()
        def highlight_values(val):
            if val > limites:
                return "background-color: red; color: white;"
            else:
                return "background-color: green; color: white;"

        styled_y = y.style.map(highlight_values, subset=["Componente Transversal (kt)"])

        return y, styled_y

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
        return datos_filtrados1.iloc[0].round(2)

    def tabla_grafico(self,tabla):
        tabla_para_grafico=tabla
        total_datos_orgin=len(tabla_para_grafico)
        
        tabla_para_grafico['Intensidad (kt)'] = pd.to_numeric(tabla_para_grafico['Intensidad (kt)'], errors='coerce')

        intervalos_nudos = [-0.1, 10] + list(np.arange(20, tabla_para_grafico['Intensidad (kt)'].max() + 10, 10))
        tabla_para_grafico['Direccion'] = self.redondear_personalizado(tabla_para_grafico['Direccion'])
        tabla_para_grafico["Direccion"]= tabla_para_grafico["Direccion"].replace(0, 360)
        
        tabla_para_grafico.sort_values(by='Direccion', inplace=True)
        tabla_para_grafico['Intensidad (kt)'] = pd.cut(tabla_para_grafico['Intensidad (kt)'], bins=intervalos_nudos, right=True)

        tabla_para_grafico['Intensidad (kt)'] = tabla_para_grafico['Intensidad (kt)'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
        tabla_para_grafico['Direccion'] = ((tabla_para_grafico['Direccion'] - 1) // 10 * 10 + 10).astype(int)

        tabla_para_grafico = tabla_para_grafico.groupby(['Direccion', 'Intensidad (kt)'], observed=False).size().unstack(fill_value=0)

        for col in tabla_para_grafico.columns:
            tabla_para_grafico[col] = pd.to_numeric(tabla_para_grafico[col], errors='coerce').fillna(0)

    # Realizar el cálculo de porcentaje
        tabla_para_grafico_1 = (tabla_para_grafico / total_datos_orgin) * 100

        return tabla_para_grafico_1.round(2), tabla_para_grafico
    
    ###################################### GRAFICOS ######################################

    def grafico_principal(self,dir_pista,limite,df_graf1):
    # Crear la figura y el eje polar
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3, 3))

        # Direcciones en radianes
        direcciones = np.radians(df_graf1.index)  # Convertir grados a radianes
        
        # Iterar sobre las columnas del DataFrame
        suma_primera_columna = df_graf1.iloc[:, 0].sum() 
        suma_redondeada = round(suma_primera_columna,2)
        ax.text(0, 0, f"{suma_redondeada}%", fontsize=5, ha='center', va='center', fontweight='bold', color='blue')
        for i in range(1, df_graf1.shape[1]):
            # Obtener el radio y los valores correspondientes
            radio = (i) * 10
            valores = df_graf1.iloc[:, i]
            
            # Agregar los textos
            for j, valor in enumerate(valores):
                if valor > 0.1:
                    angulo_grados = np.degrees(direcciones[j])  # Convertir de radianes a grados
                    posicion_radial = radio + 5
                    rotacion = angulo_grados if angulo_grados <= 180 else angulo_grados - 360 
                    if angulo_grados == 180:  # Para 0º (Norte) y 180º (Sur), texto vertical
                        rotacion = 90
                    elif angulo_grados == 360:  # Para 0º (Norte) y 180º (Sur), texto vertical
                        rotacion = 270
                    elif angulo_grados == 90 or angulo_grados == 270:  # Para 90º (Este) y 270º (Oeste), texto horizontal
                        rotacion = 0
                    elif 290 <= angulo_grados < 300:
                        rotacion +=50
                    elif 230 <= angulo_grados < 240:
                        rotacion +=180
                    elif 250 <= angulo_grados < 260:
                        rotacion +=120
                    elif 200 <= angulo_grados < 210:
                        rotacion +=220
                    elif 340 <= angulo_grados < 350:
                        rotacion +=300
                    elif 70 <= angulo_grados < 80:
                        rotacion +=300
                    elif 110 <= angulo_grados < 120:
                        rotacion +=220
                    elif 140 <= angulo_grados < 150:
                        rotacion +=180
                    elif 160 <= angulo_grados < 170:
                        rotacion +=140
                    elif 20 <= angulo_grados < 30:
                        rotacion +=40
                    elif 320 <= angulo_grados < 330:
                        rotacion +=0
                    elif 350 <= angulo_grados < 360:
                        rotacion +=300
                    elif 10 <= angulo_grados < 20:
                        rotacion +=70
                
                    ax.text(
                        direcciones[j],
                        posicion_radial,  # Desplazar el texto ligeramente afuera
                        f"{valor:.2f}%",
                        fontsize=3.3,
                        rotation=rotacion,  # Mantener el texto vertical
                        ha='center',
                        va='center',
                        rotation_mode='anchor' ,
                        fontweight='bold',  # Texto en negrita
                        color='blue'
                    )

        # Agregar línea para la dirección de la pista
        angulo_pista_rad = np.radians(dir_pista)
    
    # Trazar la línea opuesta de la pista
        angulo_opuesto_rad = np.radians((dir_pista + 180) % 360)  # Dirección opuesta (180 grados)
        grosor = 3.5 * (limite - 10) + 35

        ax.plot([angulo_pista_rad, angulo_opuesto_rad], [40, 50], color='green', lw=grosor, alpha=0.3)
        ax.plot([angulo_pista_rad, angulo_opuesto_rad], [50, 50], linestyle='--', color='green', lw=1, alpha=0.7)
        legend_line = Line2D([0], [0], color='green', lw=2, alpha=0.6)

        # Configurar ejes
        ax.set_theta_zero_location("N")  # Norte en la parte superior
        ax.set_theta_direction(-1)  # Dirección en sentido horario
        ax.set_xticks(np.radians(range(0, 360, 10)))
        ax.set_yticks(range(10, 51, 10))
        ax.set_ylim(0, 50)

        labels = ['360' if i == 0 else str(i) for i in range(0, 360, 10)] 
        ax.set_xticklabels(labels)

        ax.tick_params(axis='y', colors='black',labelsize=3)
        plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=3)

        ax.tick_params(axis='x', labelsize=5)
        ax.grid(True)

        # Títulos y leyenda
        ax.set_title("Rosa de los Vientos", va='bottom',fontsize=10)
        ax.legend([legend_line], [f"Pista {dir_pista}°"], loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=5)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Cerrar la figura para liberar memoria

        # Mostrar el gráfico en Streamlit
        st.image(buf)

    def plot_bar(self,tabla): 
        fig, ax = plt.subplots() 
        tabla.plot(kind='bar', ax=ax, figsize=(12, 6)) 
        ax.set_title('Frecuencia de Vientos por Intervalos de Nudos') 
        ax.set_xlabel('Dirección (grados)') 
        ax.set_ylabel('Frecuencia') 
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Cerrar la figura para liberar memoria
        st.image(buf)

    def plot_line(self,tabla): 
        fig, ax = plt.subplots() 
        tabla.plot(kind='line', ax=ax, figsize=(12, 6)) 
        ax.set_title('Frecuencia de Vientos por Intervalos de Nudos') 
        ax.set_xlabel('Dirección (grados)') 
        ax.set_ylabel('Frecuencia') 
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Cerrar la figura para liberar memoria
        st.image(buf)

    def plot_scatter(self,tabla): 
        fig, ax = plt.subplots() 
        for column in tabla.columns: 
            ax.scatter(tabla.index, tabla[column],label=column)
        #for column in tabla.columns: ax.scatter(tabla.index, tabla[column], label=column) 
        ax.set_title('Frecuencia de Vientos por Intervalos de Nudos') 
        ax.set_xlabel('Dirección (grados)') 
        ax.set_ylabel('Frecuencia') 
        ax.legend(title='Intervalos de Nudos', fontsize=5,title_fontsize=7) 
        ax.set_xticks(tabla.index)
        ax.tick_params(axis='x', rotation=90)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Cerrar la figura para liberar memoria
        st.image(buf)

    def plot_polar(self,tabla): 
        angles = np.deg2rad(tabla.index.values)
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
        
        # Crear colores dinámicamente para todas las columnas
        colors = plt.cm.viridis(np.linspace(0, 1, len(tabla.columns)))
        
        # Ajustar el ancho de los pétalos
        width = (2 * np.pi) / len(tabla.index)
        max_radii = np.max(tabla.values)
        
        # Verificar las columnas y sus colores
        for col, color in zip(tabla.columns, colors):
            radii = tabla[col].values
            ax.bar(angles, radii, width=width, edgecolor=color, color=color, alpha=0.8, label=col)
        
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_thetagrids(range(0, 360, 10), labels=[f"{i}" if i != 0 else "360" for i in range(0, 360, 10)],fontsize=4)
        
        # Asegurar que los círculos concéntricos se muestren correctamente hasta el máximo valor de nudos
        #rticks = np.arange(0, max_radii)
        #ax.set_rticks(rticks)
        ax.set_ylim(0, max_radii)
        ax.set_rticks([])
        ax.set_rlabel_position(-22.5)
        
        ax.legend(title='Intervalos de Nudos', fontsize=5, title_fontsize=5, bbox_to_anchor=(1.5, 1), loc='upper right')
        ax.set_title('Frecuencia de Vientos por Dirección y Nudos', fontsize=5)
        
        # Guardar la figura en un buffer y mostrarla con Streamlit
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Cerrar la figura para liberar memoria
        st.image(buf)

    def grafico1(self,tablita_df,pista,limite):
        fig = go.Figure()
       
        angulo_pista = pista  # Dirección ingresada por el usuario
        angulo_opuesto = (angulo_pista + 180) % 360  # Dirección opuesta

        limite_knots = {
            (10, 11): 107,
            (12, 13): 125,
            (14, 20): 210
        }

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
        tickvals = [0,10, 20, 30, 40, 50]
        #tablita_df = tablita_df.to_frame().T 
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
                        marker=dict(symbol='circle', color="black", size=4),
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

#####################################MOSTRAR RESULTADOS EN LA APP####################################

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
otros_graficos=st.container()

with header_container:

    page = st.selectbox("Selecciona una opción", ["Ver Resultados", "Pruebas individuales","Otros Gráficos"])

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
                #pa_el_box=resultados.nu_tabla_deca(dir_pista,copia)
                pa_el_box, comp_transv_styled = resultados.nu_tabla_deca(dir_pista,copia)

                st.expander("Tabla Procesada").dataframe(comp_transv_styled)
            
            cohe=resultados.coheficiente(limites,pa_el_box)
            if cohe is not None:
                st.markdown(f"**Coeficiente de utilización:** {cohe}%")
       
            df_graf,df_graf_otro=resultados.tabla_grafico(copia)
            #st.dataframe(df_graf)
            with st.expander("GRAFICO"):
                resultados.grafico_principal(dir_pista,limites,df_graf)
            #st.dataframe(df_graf)
            #st.dataframe(df_graf_otro)
                              

    if page=="Pruebas individuales": 
        with results_container:          
    
            st.subheader("Pruebas individuales")
            col1, col2 = st.columns(2)

            with col1:
                tabla_original["Direccion"] = pd.to_numeric(tabla_original["Direccion"], errors='coerce')
                tabla_original["Intensidad (kt)"] = pd.to_numeric(tabla_original["Intensidad (kt)"], errors='coerce')

                grid_options = GridOptionsBuilder.from_dataframe(tabla_original)
                grid_options.configure_selection('single')  # Permite seleccionar una fila

# Renderizamos la tabla con Ag-Grid
                grid_response = AgGrid(tabla_original, gridOptions=grid_options.build())

# Extraemos el índice de la fila seleccionada
                fila_seleccionada_aggrid = grid_response['selected_rows']

                #st.expander("Tabla Original").dataframe(grid_response)
                if fila_seleccionada_aggrid is not None and len(fila_seleccionada_aggrid) > 0:
                    
                    #fila_seleccionada = fila_seleccionada_aggrid[0]
                    direccion = fila_seleccionada_aggrid['Direccion'].values[0]
                    intensidad_kt = fila_seleccionada_aggrid['Intensidad (kt)'].values[0]
    
    # Realiza el cálculo
                    diferencia = direccion - dir_pista
                    dif_rad = np.radians(diferencia)
                    seno = np.sin(dif_rad)
                    seno = np.abs(seno)
                    y = seno * intensidad_kt  # Aquí 'y' será un número único
                    with col2:
                        st.write(fila_seleccionada_aggrid)
                        
    
                        st.write(f"Componente Transversal (kt): {y.round(3)}")
    
    # Comparación con 'limites'
                        if y <= limites:
                            st.write("Coeficiente de utilización: 100%")
                        else:
                            st.write("Coeficiente de utilización: 0")
                    with st.expander("GRAFICO_1"):
                        resultados.grafico1(fila_seleccionada_aggrid,dir_pista,limites)
    
    if page=="Otros Gráficos":
        df_graf,df_graf_otro=resultados.tabla_grafico(copia)

        with otros_graficos:
            st.subheader("Gráficos")
            with st.expander("Gráfico Polar"):
                resultados.plot_polar(df_graf)
            with st.expander("Gráfico de Barras"):
                resultados.plot_bar(df_graf_otro)
            with st.expander("Gráfico de Líneas"):
                resultados.plot_line(df_graf_otro)
            with st.expander("Gráfico de Dispersión"):
                resultados.plot_scatter(df_graf_otro)


                