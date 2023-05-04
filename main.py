import pandas as pd
import pymongo

# Conexión a base de datos
client = pymongo.MongoClient('mongodb+srv://hnarvaez:hnarvaez@cluster0.z7brgwz.mongodb.net/log?retryWrites=true&w=majority')
db = client.dataF1

# Obtención de datos
coleccionRe = "results"
coleccion = db[coleccionRe]
dfResults = pd.DataFrame(list(coleccion.find()))
print(dfResults)
print(dfResults.shape)
print(dfResults.columns)

# Limpieza de datos
print(dfResults.isnull().sum()) #No existen datos nulos

dfResults.drop('positionText', axis=1, inplace=True)
print(dfResults.columns)