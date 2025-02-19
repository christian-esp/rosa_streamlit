import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    h1 {
        margin-top: -70px;  /* Ajusta este valor */
    }
    </style>
    """,
    unsafe_allow_html=True
)

class ArchivoInvalidoError(Exception):
    """Excepción levantada para errores de datos en el archivo."""
    pass

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
        self.la_tabla=None

    def cargar_archivo(self, archivo):
        """Carga un archivo y verifica su validez."""
        if archivo is None:
            st.error("No se seleccionó ningún archivo.")
            return

        try:
            self.df_final = self.read_file(archivo)
            self.verificar()  # Verificar el archivo
            
            # Si la verificación es exitosa, aplicar modificaciones
            self.df_final.columns = ['Direccion', 'Intensidad (kt)']
            #self.df_final['Direccion'] = self.df_final['Direccion'].replace(0, 360)
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
        
        df_nu_tabla1=self.df_final.copy()
        total_datos=len(df_nu_tabla1)
        df_nu_tabla1.iloc[:, 0] = df_nu_tabla1.iloc[:, 0].where(df_nu_tabla1.iloc[:, 0] != 0, 360)
        df_nu_tabla1.columns = ['Direccion', 'Intensidad (kt)']

        diferencia=df_nu_tabla1["Direccion"] - pista
        dif_rad= np.radians(diferencia)  # Resta del índice al valor
        seno = np.sin(dif_rad) 
        seno=np.abs(seno) 
        y=seno*df_nu_tabla1["Intensidad (kt)"]
        comp_transv=pd.DataFrame(y,columns=["Intensidad\nComponente transversal"])
        self.la_tabla = pd.concat([df_nu_tabla1, comp_transv], axis=1)

        self.frec_con_limit=len(self.la_tabla[self.la_tabla["Intensidad\nComponente transversal"]<=limite])
        self.coheficiente_nu=round((self.frec_con_limit/total_datos)*100,2)

        conversion=self.la_tabla["Intensidad\nComponente transversal"]*1.852
        
        self.la_tabla["intensidad Componente Transversal (km/h)"] = conversion

        return self.la_tabla
    
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
        self.la_tabla_deca = pd.concat([df_nu_tabla, comp_transv], axis=1)

        self.frec_con_limit=len(self.la_tabla_deca[self.la_tabla_deca["Intensidad\nComponente transversal"]<=limite])
        self.coheficiente_nu=round((self.frec_con_limit/total_datos)*100,2)

        conversion=self.la_tabla_deca["Intensidad\nComponente transversal"]*1.852
        
        self.la_tabla_deca["intensidad Componente Transversal (km/h)"] = conversion

        return self.la_tabla_deca

    def agrupar(self, intervalo):
        if self.df_final is None:
            st.error("Debe cargar un archivo primero.")
            return
        elif not self.verificacion_exitosa:
            st.error("El archivo cargado no cumple con el formato deseado.")
            return
        df_agrupar=self.df_final.copy()
        conteo_viento_calma = df_agrupar[df_agrupar["Intensidad (kt)"] == 0].shape[0]
        conteo_total = df_agrupar[df_agrupar["Intensidad (kt)"] > 0].shape[0]

        self.viento_calma = conteo_viento_calma
        self.suma_total_frec = conteo_total

        intervalos_nudos = np.arange(0, df_agrupar['Intensidad (kt)'].max() + intervalo, intervalo)
        
        df_agrupar['Direccion'] = self.redondear_personalizado(df_agrupar['Direccion'])
        
        df_agrupar.iloc[:, 0] = df_agrupar.iloc[:, 0].where(df_agrupar.iloc[:, 0] != 0, 360)
        df_agrupar.sort_values(by='Direccion', inplace=True)
        df_agrupar['Intervalo_Nudos'] = pd.cut(df_agrupar['Intensidad (kt)'], bins=intervalos_nudos, right=True)

        df_agrupar['Intervalo_Nudos'] = df_agrupar['Intervalo_Nudos'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
        self.ordenado = df_agrupar.groupby(['Direccion', 'Intervalo_Nudos'], observed=False).size().unstack(fill_value=0)
        self.ordenado = self.ordenado.apply(pd.to_numeric, errors='coerce')
        self.ordenado = self.ordenado.applymap(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)
        self.viento_calma_fila = pd.DataFrame([[self.viento_calma]], columns=['Viento Calma'])
        self.viento_calma_fila = self.viento_calma_fila.apply(pd.to_numeric, errors='coerce')
        self.viento_calma_fila = self.viento_calma_fila.applymap(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)
        
        return self.ordenado

    def redondear_personalizado(self, numeros):
        parte_decimal = numeros - np.floor(numeros)
        return np.where(parte_decimal >= 0.5, np.ceil(numeros), np.floor(numeros))
    
    def df_vientos_cruzado(self,valor):
    
        etiquetas=self.ordenado.columns

        extremos_derechos = [int(col.split('-')[1]) for col in self.ordenado.columns]
        self.v_c = pd.DataFrame(index=self.ordenado.index, columns=etiquetas)
                
        for i in self.ordenado.index:
            for j, col in enumerate(self.ordenado.columns):
                diferencia = i - valor
                diferencia_rad = np.radians(diferencia)  # Resta del índice al valor
                resultado = np.sin(diferencia_rad) * extremos_derechos[j]
                resultado=np.abs(resultado)
                resultado=resultado.round(2)# Multiplica por el extremo derecho
                self.v_c.at[i, col] = resultado

        return self.v_c
    
    def frec_con_limi(self,limite):
        self.f_ad_perso = pd.DataFrame(index=self.v_c.index, columns=self.v_c.columns)
        
        def limite_pers(frec,valora):
            if 0<=limite<=40:
                return frec if valora <= limite else 0
                
        for col_name in self.v_c.columns:
            for index in self.v_c.index:
                self.f_ad_perso.at[index, col_name] = limite_pers(self.ordenado.at[index, col_name], self.v_c.at[index,col_name])
        self.f_ad_perso = self.f_ad_perso.apply(pd.to_numeric, errors='coerce')
        self.suma_f_ad_perso=self.f_ad_perso.sum().sum()

        self.f_ad_perso = self.f_ad_perso.map(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)
              
        self.coheficiente = round((self.suma_f_ad_perso + self.viento_calma) / (self.suma_total_frec + self.viento_calma) * 100, 2)
        
        return self.f_ad_perso
    
    def tabla_grafico(self):
        df_agrupar1=self.la_tabla.copy()

        intervalos_nudos = [-0.1, 10] + list(np.arange(20, df_agrupar1['Intensidad (kt)'].max() + 10, 10))
        df_agrupar1['Direccion'] = self.redondear_personalizado(df_agrupar1['Direccion'])
        
        #df_agrupar1.iloc[:, 0] = df_agrupar1.iloc[:, 0].where(df_agrupar1.iloc[:, 0] != 0, 36)
        df_agrupar1.sort_values(by='Direccion', inplace=True)
        df_agrupar1['Intensidad (kt)'] = pd.cut(df_agrupar1['Intensidad (kt)'], bins=intervalos_nudos, right=True)

        df_agrupar1['Intensidad (kt)'] = df_agrupar1['Intensidad (kt)'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
        df_agrupar1['Direccion'] = ((df_agrupar1['Direccion'] - 1) // 10 * 10 + 10).astype(int)

        df_agrupar1 = df_agrupar1.groupby(['Direccion', 'Intensidad (kt)']).size().unstack(fill_value=0)

        for col in df_agrupar1.columns:
            df_agrupar1[col] = pd.to_numeric(df_agrupar1[col], errors='coerce').fillna(0)

    # Convertir a float para operaciones de porcentaje
        df_agrupar1 = df_agrupar1.astype(float)
        suma_con_ceros=len(self.df_final)

    # Realizar el cálculo de porcentaje
        df_agrupar1 = (df_agrupar1 / suma_con_ceros) * 100

    # Redondear a 2 decimales
        df_agrupar1 = df_agrupar1.round(2)

        return df_agrupar1
         
        

class App:
    def __init__(self):
        self.resultados=ProcesadorDeDatos()
        self.dir_pista = None
        self.intervalos=None
        self.limites=None
    

    def grafico(self):
        fig = go.Figure()
        df_graf=self.resultados.tabla_grafico()

        # Agregar una línea que cruza desde la dirección de la pista hasta la dirección opuesta
        angulo_pista = self.dir_pista  # Dirección ingresada por el usuario
        angulo_opuesto = (angulo_pista + 180) % 360  # Dirección opuesta

        limite_knots = {
            (10, 11): 107,
            (12, 13): 125,
            (14, 20): 210
        }

        # Línea que va de la dirección de la pista a su dirección opuesta
        for (lower_limit, upper_limit), dist in limite_knots.items():
            if lower_limit <= self.limites <= upper_limit:
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
            if lower_limit <= self.limites <= upper_limit:
   
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
                elif 0<=valor<0.1:
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
    
    def mostrar_widgets1(self):
        with st.container():
            uploaded_file = st.sidebar.file_uploader("Seleccionar archivo Excel o CSV", type=["xlsx", "csv"], key="file_uploader_1")

        if uploaded_file is not None:
            self.resultados = ProcesadorDeDatos()
            self.resultados.cargar_archivo(uploaded_file)
            
            if self.resultados.verificacion_exitosa:
                
                with st.container():
                    self.intervalos = st.sidebar.selectbox("Seleccione intervalo (knots)", [1, 3, 5, 10],key="intervalos")
                    self.dir_pista = st.sidebar.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
                    self.limites = st.sidebar.number_input("Ingrese limites en (knots)", min_value=10, max_value=40, value=10, key="limites")
                    #self.dir_vientos=st.sidebar.number_input("direcciond el viento", min_value=1, max_value=360, value=1)
                    #self.nudo=st.sidebar.number_input("intensidad (kt)", min_value=0, max_value=360, value=0)

            # Solo muestra el botón "Agrupar" después de verificar la carga del archivo
                
                if st.button("Resultado"):
                    agrupacion = self.resultados.agrupar(self.intervalos)
                    nueva_tabla= self.resultados.nu_tabla(self.limites,self.dir_pista)
                    nueva_tabla_1=self.resultados.nu_tabla_deca(self.limites,self.dir_pista)
                
                    if agrupacion is not None:
                        viento_cruzados = self.resultados.df_vientos_cruzado(self.dir_pista)
                        frecuencias = self.resultados.frec_con_limi(self.limites)
                        suma_frecuencias = self.resultados.suma_f_ad_perso
                        viento_calma = self.resultados.viento_calma
                        conteo_total=self.resultados.suma_total_frec 
                        final = self.resultados.coheficiente         
                        tabla_grafico=self.resultados.tabla_grafico()
                        #viento_calma_nu=len(nueva_tabla[nueva_tabla["Nudos"]==0])
                        coheficiente=self.resultados.coheficiente_nu
                        frec_con_limit=self.resultados.frec_con_limit
                        st.session_state.df_agrupacion = agrupacion
                        st.session_state.df_nueva_tabla = nueva_tabla
                        st.session_state.df_nueva_tabla_1 = nueva_tabla_1
                        #la_tabla=self.resultados.la_tabla
                    # Mostrar los resultados en la interfaz
                        with st.container():
                            st.markdown(f"**Con una dirección de pista de** {self.dir_pista}° **y un limite de** {self.limites} (kt) \n\n**Coeficiente:** {coheficiente}%")
                        with st.container():
                            self.grafico() 

                        with st.expander("Mostrar Tablas"):
                            col1, col2= st.columns(2)
                            with col1:
                                g=len(nueva_tabla)
                                st.write("Componente Transversal")
                                st.write(nueva_tabla)
                                st.write("Decagrados trasnformado")
                                st.write(nueva_tabla_1)
                                
                                st.write(g)
                            with col2:
                                st.write(f"viento calma: {viento_calma}")
                              
                                agrupacion = agrupacion.apply(pd.to_numeric, errors='coerce')
                                a= agrupacion.sum().sum()
                                st.write("Frecuencias")
                                st.dataframe(agrupacion)
                                st.write(a)
                        with st.expander("Pruebas"):
                            if 'df_nueva_tabla_1' in st.session_state and st.session_state.df_nueva_tabla_1 is not None:
                                st.dataframe(nueva_tabla_1.iloc[:,0:2])
                                gb = GridOptionsBuilder.from_dataframe(nueva_tabla_1.iloc[:,0:2])
                                gb.configure_selection('single')  # Selección de una sola fila
                                grid_options = gb.build()

# Mostrar la tabla y capturar la selección
                                response = AgGrid(nueva_tabla_1.iloc[:,0:2], gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED)
                                if response['selected_rows']:
                                    seleccion = response['selected_rows'][0]  # Obtiene la fila seleccionada
                                    st.session_state.seleccionada = seleccion
                                    valor1, valor2 = seleccion['Direccion'], seleccion['Intensidad (kt)']

    # Realiza un cálculo (ej. suma) y aplica la condición
                                    calculo = valor1 + valor2
                                    resultado = 100 if calculo <= 10 else 0
                                    st.write(f"Resultado del cálculo: {calculo}")
                                    st.write(f"Condición aplicada: {resultado}")

                                else:
            # Si no se ha seleccionado ninguna fila, mostramos un mensaje o manejamos de otra forma
                                    st.write("Por favor, selecciona una fila de la tabla.")
    # Muestra el resultado en la interfaz
                            
                            
                            
                        #st.dataframe(tabla_grafico)
                        #j=tabla_grafico.iloc[:,0].sum()
                        #st.write(j)

                                            
            else:
                st.error("El archivo cargado no cumple con el formato deseado.")
        else:
            st.warning("Por favor, cargue un archivo para continuar.")

def main():
    st.title("ANALISIS DE VIENTOS")

    # Crear una instancia de la clase InterfazStreamlit
    interfaz = App()

    # Llamar al método que muestra los widgets de la interfaz
    interfaz.mostrar_widgets1()

if __name__ == "__main__":
    main()
