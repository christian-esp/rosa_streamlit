import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.lines import Line2D 
from fpdf import FPDF
import io
from matplotlib.patches import Polygon
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from windrose import WindroseAxes
import matplotlib.cm as cm
import matplotlib.image as mpimg
from PIL import Image



st.set_page_config(layout="wide",page_title="Análisis de vientos")

image_path = "images/logo.png" 

st.markdown(
    """
    <style>
    [data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    
    def nu_tabla_deca(self, pista,data_base,limites):
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
        data_base["Intensidad (km/h)"]=(data_base["Intensidad (kt)"]*1.852).round(2)
        data_base["Intensidad (km/h)"]=pd.to_numeric(data_base["Intensidad (km/h)"], errors='coerce')
        data_base["Intensidad (kt)"] = pd.to_numeric(data_base["Intensidad (kt)"], errors='coerce')
        y=seno*data_base["Intensidad (kt)"]
        yy=seno*data_base["Intensidad (km/h)"]
        y.name = "Nudos"
        yy.name="km/h"
        y=y.to_frame()
        yy=yy.to_frame()
        

        def highlight_values_nudos(val):
            if val > limite_kt:
                return "background-color: red; color: white; text-align: left;"
            else:
                return "background-color: green; color: white; text-align: left;"
        def highlight_values_km(val):
            limites_km=round(limite_kt*1.852,2)


            if val > limites_km:
                return "background-color: red; color: white;text-align: left;"
            else:
                return "background-color: green; color: white;text-align: left;"

        styled_y = y.style.map(highlight_values_nudos, subset=["Nudos"]).format("{:.2f}")
        styled_yy = yy.style.map(highlight_values_km, subset=["km/h"]).format("{:.2f}")
        

        return y,yy, styled_y,styled_yy

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
    
    def archivo_prueba(self):
        # Generar direcciones aleatorias entre 10 y 360, repitiendo algunas direcciones
        directions = np.random.choice(range(10, 361), size=100, replace=True)

# Generar intensidades aleatorias entre 0.1 y 50
        intensity = np.random.uniform(0.1, 50, size=100)

# Crear el DataFrame
        df = pd.DataFrame({
            'Direccion': directions,
            'Intensidad_Viento': intensity
        })
        df["Intensidad_Viento"]=df["Intensidad_Viento"].round(2)
        output = io.BytesIO()  # Crear un buffer en memoria
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos_Viento')
        return output.getvalue()
    
    ###################################### GRAFICOS ######################################
    def grafico_principal_prueba(self,tabla_grafico,dir_pista,limite):
        #imagen_fondo = mpimg.imread("C:/Users/cespi/Desktop/Proyecto_rosa/rosa_streamlit/logogta.png")

        frecuencias = [10, 20, 30, 40,50]
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4), dpi=300)
        ax.set_facecolor('none')
        imagen_fondo = Image.open("logogta.png")
        imagen_redimensionada = imagen_fondo.resize((250, 150))  # Cambiar a tu tamaño deseado
        imagen_redimensionada = np.array(imagen_redimensionada)

# Convertir a formato compatible con Matplotlib
        imagen_array = np.asarray(imagen_redimensionada)

# Mostrar con figimage
        fig.figimage(imagen_array, xo=260, yo=320, alpha=0.5, zorder=-1)

        ax.set_theta_zero_location("N")  # 0° en la parte superior (Norte)
        ax.set_theta_direction(-1)       # Sentido horario
        #ax.set_title("Rosa de Vientos", y=1.05, fontsize=10)
        #fig.figimage(imagen_fondo, xo=0, yo=0, alpha=0.5, zorder=-1)
# Etiquetas angulares centradas en los sectores
        angulos = np.arange(0, 360, 10) 
        ax.set_xticks(np.deg2rad(angulos + 5))  # Ajustar +5 grados para centrar etiquetas entre los rayos
        ax.set_xticklabels([" "]*len(angulos), fontsize=10)
        ax.set_rgrids(range(10, max(frecuencias) + 10, 10), angle=90, fontsize=5)

        suma_primera_columna = tabla_grafico.iloc[:, 0].sum() 
        suma_redondeada = round(suma_primera_columna,2)
        ax.text(0, 0, f"{suma_redondeada}", fontsize=7, ha='center', va='center', color='blue',zorder=5)

        manual_labels = [
            (0, "360°"), (10, "10°"), (20, "20°"), (30, "30°"), (40, "40°"), 
            (50, "50°"), (60, "60°"), (70, "70°"), (80, "80°"), (90, "90°"), 
            (100, "100°"), (110, "110°"), (120, "120°"), (130, "130°"), (140, "140°"),
            (150, "150°"), (160, "160°"), (170, "170°"), (180, "180°"), (190, "190°"), 
            (200, "200°"), (210, "210°"), (220, "220°"), (230, "230°"), (240, "240°"),
            (250, "250°"), (260, "260°"), (270, "270°"), (280, "280°"), (290, "290°"),
            (300, "300°"), (310, "310°"), (320, "320°"), (330, "330°"), (340, "340°"),
            (350, "350°")
        ]
        

        desplazamiento=0.2
        direcciones = np.deg2rad(tabla_grafico.index)
        for angle_deg, label in manual_labels:
            angle_rad = np.deg2rad(angle_deg+desplazamiento)  # Convertir a radianes
            ax.text(
                angle_rad, 1.1 * max(frecuencias),  # Coordenadas (ángulo, radio)
                label, fontsize=5, ha='center', va='center',fontweight='bold', color='#2c3e50'  # Ajustes del texto
            )

        ax.set_rlim(0, max(frecuencias) + 10)
        dir_pista_rad = np.deg2rad(dir_pista)
        grosor = 3.5 * (limite - 10) + 40.1

        angulo_opuesto_rad = np.radians((dir_pista + 180) % 360)
        ax.plot([dir_pista_rad, angulo_opuesto_rad], [0, max(frecuencias)*1.9], color='black', linewidth=1,linestyle='--', alpha=0.7,zorder=4)
        ax.plot([angulo_opuesto_rad, dir_pista_rad], [0, max(frecuencias)*1.9], color='black', linewidth=1,linestyle='--', alpha=0.7,zorder=4)
        
        if limite==10:
            ax.plot([dir_pista_rad, angulo_opuesto_rad], [50, 50], color='green', lw=grosor, alpha=0.5)
        elif limite==13:
            ax.plot([dir_pista_rad, angulo_opuesto_rad], [50, 50], color='#1abc9c', lw=grosor, alpha=0.5)
        elif limite==20:
            ax.plot([dir_pista_rad, angulo_opuesto_rad], [50, 50], color='#2ecc71', lw=grosor, alpha=0.5)

        for i in range(1, tabla_grafico.shape[1]):
            radio = (i) * 10
            valores = tabla_grafico.iloc[:, i]
            
            # Agregar los textos
            for j, valor in enumerate(valores):
                if valor > 0.1:
                    #angulo_grados = np.degrees(direcciones[j])  # Convertir de radianes a grados
                    posicion_radial = radio + 5
                    ax.text(
                        direcciones[j],
                        posicion_radial,  # Desplazar el texto ligeramente afuera
                        f"{valor:.2f}",
                        fontsize=3,
                  # Mantener el texto vertical
                        ha='center',
                        va='center',
                        rotation_mode='anchor' ,
                        fontweight='bold',  # Texto en negrita
                        color='blue'
                    )
                elif 0<valor <= 0.1 :
                    posicion_radial = radio + 5
                    ax.text(
                        direcciones[j],
                        posicion_radial,  # Desplazar el texto ligeramente afuera
                        "+",
                        fontsize=3.3,
                        ha='center',
                        va='center',
                        rotation_mode='anchor' ,
                        fontweight='bold',  # Texto en negrita
                        color='blue'
                    )


        plt.tight_layout()
        st.pyplot(fig)
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Añadir botón de descarga en Streamlit
        
        st.download_button(
                label="Descargar Gráfico",
                data=buf,
                file_name="grafico_polar.png",
                mime="image/png"
            )

        return fig

    def anemograma(self, tabla):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = WindroseAxes.from_ax(fig=fig)
        
        # Ajustar los bins para que sean solo números sin corchetes ni paréntesis
        bins = np.arange(0, 51, 10)
        labels = [f"{i}-{i+10}" for i in bins[:-1]]  # Crear las etiquetas sin corchetes ni paréntesis

        ax.contourf(tabla["Direccion"], tabla["Intensidad (kt)"], bins=bins, cmap=cm.viridis)

        ax.set_yticklabels([10, 20, 30, 40, 50], fontsize=8) 
        ax.tick_params(axis='both', labelsize=8)

        # Ajustar la leyenda para mostrar los nuevos labels y reducir el tamaño del texto
        ax.set_legend(labels=labels, fontsize=6, loc='upper left', bbox_to_anchor=(1.1, 0.8), markerscale=1.5, labelspacing=0.05)

        plt.tight_layout()
        st.pyplot(fig)
        
        fig = plt.gcf()  
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="Descargar Gráfico",
            data=buf,
            file_name="grafico_polar.png",
            mime="image/png",
            key="unique_download_button_1" 
        )
        
        return fig

    def grafico_principal1(self, dir_pista, limite, df_graf1):
    # Crear la figura y el eje polar
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3, 3))

        # Direcciones en radianes
        direcciones = np.radians(df_graf1.index)  # Convertir grados a radianes

        # Definir un rango de colores para los círculos
        color_map = plt.cm.viridis  # O cualquier otro colormap que prefieras
        max_frecuencia = df_graf1.max().max()  # Máxima frecuencia de viento
        max_intensidad = df_graf1.max(axis=0).max()  # Máxima intensidad de viento

        for i in range(0, df_graf1.shape[1]):  # Para cada columna (que representa una dirección específica)
            valores = df_graf1.iloc[:, i]
            
            # Iterar sobre cada valor de la columna
            for j, valor in enumerate(valores):
                if valor > 0:
                    # Definir el tamaño de los círculos basado en la intensidad del viento
                    radio = (i) * 10  # Ajusta esto si es necesario
                    tamaño_circulo = valor / max_frecuencia * 50  # Ajuste del tamaño de los círculos
                    color = color_map(valor / max_frecuencia)  # Asignar color según la frecuencia

                    # Modificar estilo de los círculos, zorder=3 asegura que los círculos están sobre los rayos
                    ax.scatter(
                        direcciones[j], 
                        radio + 5, 
                        s=tamaño_circulo,  # Tamaño ajustado
                        color=color,       # Color basado en la frecuencia
                        alpha=0.9,         # Transparencia
                         # Borde de los círculos en negro
                        linewidths=0.01,    # Grosor del borde
                        marker='o',        # Forma circular
                        zorder=3           # Coloca los círculos encima de los rayos
                    )

        # Agregar línea para la dirección de la pista
        angulo_pista_rad = np.radians(dir_pista)
        angulo_opuesto_rad = np.radians((dir_pista + 180) % 360)  # Dirección opuesta (180 grados)
        grosor = 3.5 * (limite - 10) + 35
        ax.plot([angulo_pista_rad, angulo_opuesto_rad], [40, 50], color='green', lw=grosor, alpha=0.3, zorder=1)  # zorder=1 para los rayos
        ax.plot([angulo_pista_rad, angulo_opuesto_rad], [50, 50], linestyle='--', color='green', lw=1, alpha=0.7, zorder=1)

        # Configurar ejes
        ax.set_theta_zero_location("N")  # Norte en la parte superior
        ax.set_theta_direction(-1)  # Dirección en sentido horario
        ax.set_xticks(np.radians(range(0, 360, 10)))
        ax.set_yticks(range(10, 51, 10))
        ax.set_ylim(0, 50)

        labels = ['360' if i == 0 else str(i) for i in range(0, 360, 10)] 
        ax.set_xticklabels(labels)
        ax.tick_params(axis='y', colors='black', labelsize=3)
        ax.tick_params(axis='x', labelsize=5)
        ax.grid(True)

        # Títulos y leyenda
        ax.set_title("Distribución de Vientos", va='bottom', fontsize=7)
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

    def grafico1(self,intensidad,angulo_viento,pista,limite,t,buf):
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
                if t<=limite:

                    fig.add_trace(go.Scatterpolar(
                        r=[radio],
                        theta=[angulo_viento],
                        mode='markers',
                        marker=dict(symbol='circle', color="green", size=4),
                        showlegend=False
                    ))
                    break 
                else:
                    fig.add_trace(go.Scatterpolar(
                        r=[radio],
                        theta=[angulo_viento],
                        mode='markers',
                        marker=dict(symbol='circle', color="red", size=4),
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
        fig.write_image(buf, format='png')
        buf.seek(0)
        st.plotly_chart(fig, config=config)


##########################################################MINFORME##################################################

    def generar_pdf(self,y,dir_pista, limite,grafico_scatter_buf):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Título
        pdf.cell(200, 10, txt="INFORME", ln=True, align='C')
        pdf.ln(10)  # Espaciado
        texto = f"COHEFICIENTE DE UTILIZACIÓN: {y} %\nDirección de Pista: {dir_pista}º \nLímite del Componente Transversal permitido: {limite}"
        pdf.multi_cell(0, 10, texto) 
        pdf.ln(10)  # Espaciado
        if grafico_scatter_buf and not grafico_scatter_buf.getvalue() == b'':
            grafico_scatter_buf.seek(0)  # Asegúrate de que el puntero esté al inicio
            pdf.image(grafico_scatter_buf, x=10, y=None, w=190, type='PNG')
        else:
            pdf.cell(0, 10, txt="No se pudo generar el gráfico.", ln=True, align='C')
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Devuelve los bytes del PDF
        return BytesIO(pdf_bytes) 

#####################################MOSTRAR RESULTADOS EN LA APP####################################
resultados = ProcesadorDeDatos()
uploaded_file = st.file_uploader("", type=["xlsx", "csv"], key="file_uploader_1")
dir_pista = st.sidebar.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
#limites = st.sidebar.number_input("Ingrese el límite de Componente Transversal (kt)", min_value=10, max_value=40, value=10, key="limites")
limites=st.sidebar.selectbox("Selecciona un límite", ["10 kt - 19 km/h", "13 kt - 24 km/h", "20 kt - 37 km/h"])
def extraer_primer_numero(texto):
    return int(texto.split()[0])
limite_kt = extraer_primer_numero(limites)

excel_data = resultados.archivo_prueba()
st.sidebar.download_button(
        label="Descargar formato",
        data=excel_data,
        file_name="archivo_prueba.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if "tabla_original" not in st.session_state:
    st.session_state["tabla_original"] = None
if "tabla_procesada" not in st.session_state:
    st.session_state["tabla_procesada"] = None
if "fila_seleccionada" not in st.session_state:
    st.session_state["fila_seleccionada"] = None

#SI APRETO EL BOTON DE CARGA DE ARCHIVO:
#resultados = ProcesadorDeDatos()
header_container = st.container()
# Contenedor para los resultados
results_container = st.container()
otros_graficos=st.container()

with header_container:

    page = st.selectbox("Selecciona una opción", ["Ver Resultados", "Pruebas individuales","Otros Gráficos"])

if uploaded_file is not None:
    tabla_original=resultados.cargar_archivo(uploaded_file)
    tabla_original["Intensidad (km/h)"]=(tabla_original["Intensidad (kt)"]*1.852).round(2)
    copia=tabla_original.copy()
    
    if page=="Ver Resultados":
        with results_container:
            st.subheader("Resultados")
            col1, col2 = st.columns(2)

            with col1:
                st.expander("Tabla Original").dataframe(tabla_original)

            with col2:
                
                pa_el_box, pa_el_box1,comp_transv_styled,comp_transv_styled1 = resultados.nu_tabla_deca(dir_pista,copia,limites)

                with st.expander("Intensidad Componente Transversal"):
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        
                        st.dataframe(comp_transv_styled)

                    with col_right:
                        
                        
                        st.dataframe(comp_transv_styled1)
            
            cohe=resultados.coheficiente(limite_kt,pa_el_box)
            if cohe is not None:
                direccion_opuesta = (dir_pista + 180) % 360
                direccion_opuesta_deca=round(direccion_opuesta/10)

                limites_km=round(limite_kt*1.852)
                dir_pista_deca=round(dir_pista / 10)

                if cohe >= 95 and dir_pista>=10 and direccion_opuesta>=10:
                    
                    st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:green;">{cohe}%</span> - Dirección de Pista: <span style="color:green;">{dir_pista_deca} - {direccion_opuesta_deca}</span>- Límite Componente Transversal: <span style="color:green;">{limite_kt} kt</span> / <span style="color:green;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)

                elif cohe >= 95 and dir_pista>=10 and direccion_opuesta<10:
                        st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:green;">{cohe}%</span> - Dirección de Pista: <span style="color:green;">{dir_pista_deca} - 0{direccion_opuesta}</span>- Límite Componente Transversal: <span style="color:green;">{limite_kt} kt</span> / <span style="color:green;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)

                elif cohe >= 95 and dir_pista<10 and direccion_opuesta<10:
                    st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:green;">{cohe}%</span> - Dirección de Pista: <span style="color:green;">0{dir_pista} - 0{direccion_opuesta}</span>- Límite Componente Transversal: <span style="color:green;">{limite_kt} kt</span> / <span style="color:green;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)
                elif cohe >= 95 and dir_pista<10 and direccion_opuesta>=10:
                    st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:green;">{cohe}%</span> - Dirección de Pista: <span style="color:green;">0{dir_pista} - {direccion_opuesta_deca}</span>- Límite Componente Transversal: <span style="color:green;">{limite_kt} kt</span> / <span style="color:green;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)

                elif cohe < 95 and dir_pista>=10 and direccion_opuesta>=10:

                    st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:red;">{cohe}%</span> - Dirección de Pista: <span style="color:red;">{dir_pista_deca} - {direccion_opuesta_deca}</span> - Límite Componente Transversal: <span style="color:red;">{limite_kt} kt</span> / <span style="color:red;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)

                elif cohe < 95 and dir_pista>=10 and direccion_opuesta<10:
                     st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:red;">{cohe}%</span> - Dirección de Pista: <span style="color:red;">{dir_pista_deca} - 0{direccion_opuesta}</span> - Límite Componente Transversal: <span style="color:red;">{limite_kt} kt</span> / <span style="color:red;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)
                elif cohe < 95 and dir_pista<10 and direccion_opuesta<10:
                     st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:red;">{cohe}%</span> - Dirección de Pista: <span style="color:red;">0{dir_pista} - 0{direccion_opuesta}</span> - Límite Componente Transversal: <span style="color:red;">{limite_kt} kt</span> / <span style="color:red;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)
                     
                elif cohe < 95 and dir_pista<10 and direccion_opuesta>=10:
                     st.markdown(f"""
                        <p style="font-size:20px; color:blue; font-weight:bold;"> 
                        Coeficiente de utilización: <span style="color:red;">{cohe}%</span> - Dirección de Pista: <span style="color:red;">0{dir_pista} - {direccion_opuesta_deca}</span> - Límite Componente Transversal: <span style="color:red;">{limite_kt} kt</span> / <span style="color:red;">{limites_km} km/h</span>
                        </p>
                        """, unsafe_allow_html=True)
       
            df_graf,df_graf_otro=resultados.tabla_grafico(copia)
            #st.dataframe(df_graf)
            with st.expander("ROSA DE VIENTOS"):
                resultados.grafico_principal_prueba(df_graf,dir_pista,limite_kt)  # Ahora esto retorna la figura
                #st.pyplot(fig) 
                
                #resultados.grafico_principal(dir_pista,limites,df_graf)
            with st.expander("ANEMOGRAMA"):
                #resultados.grafico_principal1(dir_pista, limites, df_graf_otro)
                resultados.anemograma(tabla_original)
                #st.pyplot(fig1)
            #st.dataframe(df_graf)
            st.dataframe(df_graf_otro)
            t=df_graf_otro.sum()
            st.write(t)                             

    if page == "Pruebas individuales":
        with results_container:
            st.subheader("Pruebas individuales")
            st.expander("Tabla Original").dataframe(tabla_original)
  
            tabla_original["Direccion"] = pd.to_numeric(tabla_original["Direccion"], errors="coerce")
            tabla_original["Intensidad (kt)"] = pd.to_numeric(tabla_original["Intensidad (kt)"], errors="coerce")
            num_filas=len(tabla_original)
            fila = st.number_input("Indique el índice de la fila", min_value=0, max_value=num_filas - 1, value=0, key="fila")
            try:
                dir=tabla_original.iloc[fila,0]
                nudo=tabla_original.iloc[fila,1]
                km=tabla_original.iloc[fila,2]
                diferencia=dir - dir_pista
                dif_rad= np.radians(diferencia)  # Resta del índice al valor
                seno = np.sin(dif_rad) 
                seno=np.abs(seno) 
                y=seno*nudo
                yy=seno*km
                y=y.round(2)
                yy=yy.round(2)
               
                st.write(f"Dirección del viento: {dir}º")
                st.write(f"Intensidad del viento: {nudo} kt - {km} km/h")
                st.write(f"Dirección de Pista: {dir_pista}º")
                st.write(f"Intensidad del Componente Transversal: {y} kt - {yy} km/h")
                
                if y <= limite_kt:
                    st.write("Coheficiente: 100%")
                else:
                    st.write("Coheficiente: 0")
                st.expander("Gráfico")
                buf = BytesIO()
                resultados.grafico1(nudo,dir,dir_pista,limite_kt,y,buf)
                buf.seek(0)  # Volver al inicio del buffer
                #st.image(buf, caption="Gráfico generado", use_column_width=True)

            # Crear botón para descargar la imagen
                st.download_button(
                    label="Descargar gráfico",
                    data=buf,
                    file_name="grafico_generado.png",
                    mime="image/png"
                )

            except IndexError:
                st.error("El índice seleccionado está fuera de rango.")

 
    if page=="Otros Gráficos":
        df_graf,df_graf_otro=resultados.tabla_grafico(copia)

        with otros_graficos:
            st.subheader("Gráficos")
            #with st.expander("Gráfico Polar"):
                #resultados.plot_polar(df_graf)
            with st.expander("Gráfico de Barras"):
                resultados.plot_bar(df_graf_otro)
            with st.expander("Gráfico de Líneas"):
                resultados.plot_line(df_graf_otro)
            with st.expander("Gráfico de Dispersión"):
                resultados.plot_scatter(df_graf_otro)