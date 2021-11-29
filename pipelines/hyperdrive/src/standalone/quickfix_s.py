import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/MAIS00110.csv", parse_dates=['DateTime'], index_col=0)
data.head()

# count les valeur à 0
(data['Consumption'] == 0). sum()

# pas obligé mais juste pour être consitant avec l'autre notebook
df= data.copy()[data.index < "2014-03-02 00:00:00"]
(df['Consumption'] == 0). sum()
#on remplace les 0 par NaN
df.replace(0,np.nan,inplace=True)
#semble marcher on lance une interpolation selon l'index
df.interpolate(method='time',inplace=True)
#fichier avec données interpolé (pour le fun on enregistre mais tout peut evidemment etre fait online)
df.to_csv('MAIS00110_I.csv')
