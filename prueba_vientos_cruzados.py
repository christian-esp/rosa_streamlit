import pandas as pd
import numpy as np

x=pd.read_excel("C:/Users/Invitado GTA-GIAI/Desktop/rosa_streamlit/jhon_2colum.xlsx")

x.iloc[:, 0] = x.iloc[:, 0].where(x.iloc[:, 0] != 0, 360)
x.columns = ['Direccion', 'Nudos']

x["diferencias"]=x["Direccion"] - 180
x["radianes_dif"]= np.radians(x["diferencias"])  # Resta del índice al valor
x["senodeladiferencia"] = np.sin(x["radianes_dif"]) 
x["senodeladiferencia"]=np.abs(x["senodeladiferencia"]) 
s=338.47-180
y=np.radians(s)
t=np.abs(y)
print(np.sin(t))

print(x)
