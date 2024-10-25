import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

    def cargar_archivo(self, archivo):
        if archivo is not None:
            try:
                if archivo.name.endswith('.xlsx'):
                    self.df_final = pd.read_excel(archivo)
                elif archivo.name.endswith('.csv'):
                    self.df_final = pd.read_csv(archivo, sep=None, engine='python')

            # Verificar el archivo
                self.verificar()  # Llama al método verificar
            
                self.df_final.columns = ['Direccion', 'Nudos']
                self.df_final.iloc[:, 0] = self.df_final.iloc[:, 0].where(self.df_final.iloc[:, 0] != 0, 360)
                self.verificacion_exitosa = True  # Establece como True solo si pasa la verificación
                st.success("El archivo se procesó correctamente.")
            except ArchivoInvalidoError as e:
                st.error(f"Error en los Datos: {str(e)}")
                self.verificacion_exitosa = False  # Asegúrate de establecer esto como False en caso de error
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {str(e)}")
                self.verificacion_exitosa = False  # Asegúrate de establecer esto como False en caso de error
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
        self.ordenado = self.ordenado.map(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)

        return self.ordenado

    def redondear_personalizado(self, numeros):
        parte_decimal = numeros - np.floor(numeros)
        return np.where(parte_decimal >= 0.5, np.ceil(numeros), np.floor(numeros))
    
    def df_vientos_cruzado(self,valor):
    
        etiquetas=self.ordenado.columns
        try:
            rangos = np.array([list(map(int, col.split('-'))) for col in etiquetas])
        except ValueError as e:
            st.error("Error", f"Error al procesar las etiquetas: {e}")
            return
        
        fines = rangos[:, 1]
        rad = np.pi / 180
        
        self.v_c = pd.DataFrame(index=self.ordenado.index, columns=etiquetas)
                
        dif_angulos = (self.dir_array[:, None] - valor) * rad
        
        resultados = np.abs(fines * np.sin(dif_angulos))
        resultados = np.round(resultados, 2)
        
        self.v_c.iloc[:, :] = resultados
        
        lista_valores = []
        for index, row in self.v_c.iterrows():
            for col in self.v_c.columns:
                valor = row[col]
            # Verificar si el valor es mayor a 10
                if valor > 10:
                    lista_valores.append((index, col, valor))  
        #print(self.v_c)

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

        intervalos_10 = np.arange(0, self.df_final['Nudos'].max() + 10, 10)
        
        self.df_final['Direccion'] = self.redondear_personalizado(self.df_final['Direccion'])
        
        self.df_final.iloc[:, 0] = self.df_final.iloc[:, 0].where(self.df_final.iloc[:, 0] != 0, 360)
        self.df_final.sort_values(by='Direccion', inplace=True)
        self.df_final['Direccion'] = pd.cut(self.df_final['Direccion'], bins=self.dir_graf, right=True, include_lowest=True)
        self.df_final['Direccion'] = self.df_final['Direccion'].apply(lambda x: x.right)
        self.df_final['Intervalo_Nudos'] = pd.cut(self.df_final['Nudos'], bins=intervalos_10, right=False)
        self.df_final['Intervalo_Nudos'] = self.df_final['Intervalo_Nudos'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")

        self.df_final = self.df_final.groupby(['Direccion', 'Intervalo_Nudos'], observed=False).size().unstack(fill_value=0)
        self.df_final.index=self.dir_graf
        self.df_final = self.df_final.map(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)
        self.array_df_final=self.df_final.to_numpy()
        #porcentaje = (self.array_df_final / self.suma_total_frec) * 100
        #self.tabla_porcentaje = pd.DataFrame(porcentaje, index=self.ordenado.index, columns=labels_tabla_grafico)
        #self.tabla_porcentaje=self.tabla_porcentaje.round(2)

        #self.tabla_graf = self.tabla_graf.apply(pd.to_numeric, errors='coerce')
        
        #self.suma_tabla_graf=self.tabla_graf.sum().sum()
        
        #self.tabla_graf = self.tabla_graf.map(lambda x: f'{x:.0f}' if isinstance(x, (int, float)) else x)

        #self.array_tabla_graf = self.tabla_graf.to_numpy()
        #self.tipo=type(self.array_tabla_graf)
        #porcentajes = (array_tabla_graf / self.suma_total_frec) * 100
    
        #self.tabla_porc = pd.DataFrame(porcentajes, index=self.ordenado.index, columns=self.tabla_graf.columns)
        #self.tabla_porc=self.tabla_porc.round(2)
        
        return self.df_final


class App:
    def __init__(self):
        self.resultados=ProcesadorDeDatos()
        self.dir_pista = None
        self.intervalos=None
        self.limites=None
    

    def grafico(self):
        fig = go.Figure()

        # Agregar una línea que cruza desde la dirección de la pista hasta la dirección opuesta
        angulo_pista = self.dir_pista  # Dirección ingresada por el usuario
        angulo_opuesto = (angulo_pista + 180) % 360  # Dirección opuesta

        # Línea que va de la dirección de la pista a su dirección opuesta
        fig.add_trace(go.Scatterpolar(
            r=[50, 50],  # Se extiende desde el borde interior (10) hasta el borde exterior (50)
            theta=[angulo_pista, angulo_opuesto],  # Dirección de la pista hasta la opuesta
            mode='lines',
            line=dict(color="red", width=2),
            name="Desde Pista a Opuesto"
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
                    title="Intensidad",
                    range=[0, 50],  # Ajusta el rango según tus datos
                    tickvals=[10, 20, 30, 40, 50],
                )
            ),
            title="Gráfico Polar de Direcciones de Viento",
            width=700,
            height=700
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)


    def mostrar_widgets(self):

        uploaded_file = st.file_uploader("Seleccionar archivo Excel o CSV", type=["xlsx", "csv"], key="file_uploader_1")
    
        if uploaded_file is not None:
            self.resultados = ProcesadorDeDatos()
            self.resultados.cargar_archivo(uploaded_file)

            if self.resultados.verificacion_exitosa:
                self.intervalos = st.selectbox("Seleccione intervalo (knots)", [1, 3, 5, 10],key="intervalos")
                self.dir_pista = st.number_input("Ingrese la dirección de la pista", min_value=1, max_value=360, value=1,key="dir_pista")
                self.limites = st.number_input("Ingrese limites en (knots)", min_value=10, max_value=40, value=10, key="limites")

            # Solo muestra el botón "Agrupar" después de verificar la carga del archivo
                if st.button("Resultado"):
                    agrupacion = self.resultados.agrupar(self.intervalos)
                    if agrupacion is not None:
                        viento_cruzados = self.resultados.df_vientos_cruzado(self.dir_pista)
                        frecuencias = self.resultados.frec_con_limi(self.limites)
                        suma_frecuencias = self.resultados.suma_f_ad_perso
                        final = self.resultados.coheficiente
                        tipo=self.resultados.tipo
                        tabla_grafico=self.resultados.tabla_grafico()
                        #array=self.resultados.array_tabla_graf
                        #suma_tabla=self.resultados.suma_tabla_graf
                        #tabla_porcentaje=self.resultados.tabla_porc
                        self.grafico()

                    # Mostrar los resultados en la interfaz
                        st.markdown(f"**Con una dirección de pista de** {self.dir_pista}° **y un limite de** {self.limites} knots\n\n**Coeficiente:** {final}%\n\n**Total frecuencias:** {suma_frecuencias}")
                        st.write("tipo")
                        st.write(tabla_grafico)
                        #st.dataframe(tabla_grafico)
                        #st.write(f"suma:{suma_tabla}")
                        #st.write("Resultados de Agrupación:")
                        #st.dataframe(agrupacion)
                        #st.write("Resultados de Vientos Cruzados:")
                        #st.dataframe(viento_cruzados)
                        #st.write("Frecuencias con límites:")
                        #st.dataframe(frecuencias)
                        
            else:
                st.error("El archivo cargado no cumple con el formato deseado.")
        else:
            st.warning("Por favor, cargue un archivo para continuar.")

def main():
    st.title("ROSA DE LOS VIENTOS")

    # Crear una instancia de la clase InterfazStreamlit
    interfaz = App()

    # Llamar al método que muestra los widgets de la interfaz
    interfaz.mostrar_widgets()

if __name__ == "__main__":
    main()