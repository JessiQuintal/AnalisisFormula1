import streamlit as st
import pandas as pd
import pymongo

st.title('Limpieza de datos')

# Conexión a base de datos
client = pymongo.MongoClient('mongodb+srv://hnarvaez:hnarvaez@cluster0.z7brgwz.mongodb.net/log?retryWrites=true&w=majority')
db = client.dataF1

# importar resultados luego del 2012 (agregacion MongoDB)
coleccionRe = "resultsAfter2012"
coleccion = db[coleccionRe]
results = pd.DataFrame(list(coleccion.find()))
#Visualización en st
st.subheader('Resultados luego de 2012')
st.write(results)

# importar resultados de constructores
coleccionCs = "constructorStandings"
coleccion = db[coleccionCs]
constructorStandings = pd.DataFrame(list(coleccion.find()))
# Visualización en st
st.subheader('Resultados de constructores (equipos)')
st.write(constructorStandings)

# importar resultados de pilotos
coleccionDs = "driverStandings"
coleccion = db[coleccionDs]
# Visualización en st
driverStandings = pd.DataFrame(list(coleccion.find()))
st.subheader('Resultados de pilotos')
st.write(driverStandings)

#Eliminamos columnas que no nos sirven
# Agarramos solo las columnas que necesitamos
results = results[["raceId","driverId","constructorId", "grid", "position"]]
results[["raceId","driverId","constructorId","grid"]]=results[["raceId", "driverId","constructorId", "grid"]].astype(int)
# Visualización en st
st.subheader('Eliminamos columnas que no nos sirven')
st.write('Resultados después de 2012')
st.write(results.head())

constructorStandings = constructorStandings[["raceId", "constructorId", "position"]].astype(int)
# Hacemos lo mismo con la columna "position"
constructorStandings = constructorStandings.rename(columns={"position": "constructorStanding"})
# Lo mismo para la situiente carrera
constructorStandings["raceId"] += 1
#Visualización en st
st.write('Resultados de constructores')
st.write(constructorStandings.head())

driverStandings = driverStandings[["raceId", "driverId", "position"]].astype(int)
# Renombramos la columna de position para evitar conflictos
driverStandings = driverStandings.rename(columns={"position": "driverStanding"})
# Aumentamos en uno para emparejar standings con resultados de carrera
driverStandings["raceId"] += 1
#Visualización st
st.write('Resultados de pilotos')
st.write(driverStandings.head())

# Unimos los resultados de carrera a los resultados de los pilotos
resultsDriverStandings = pd.merge(results, driverStandings, on=["raceId", "driverId"], how="inner")
st.subheader('Unión de resultados de carrera y resultados de piloto')
st.write(resultsDriverStandings.head())

# Unimos los resultados de carrera a los resultados de los equipos
joinedData = pd.merge(resultsDriverStandings, constructorStandings, on=["raceId", "constructorId"], how="inner")
st.subheader('Unión de resultados de carrera y resultados de equipo')
st.write(joinedData.head())
st.write(joinedData.info())

#Filtrados
st.subheader('Filtrar resultados de posiciones finales')
st.write(joinedData[["grid", "driverStanding", "constructorStanding", "position"]].agg(['min', 'max']))

# Guardamos sólo los datos que necesitamos para el posible modelo
dataset = joinedData[["driverStanding", "constructorStanding", "grid", "position"]]
st.write('Eliminamos datos')
st.write(dataset.head())

# Filtramos sólo los resultados que hayan terminado en un número de posición
dataset = dataset[dataset.position.apply(lambda x: x.isnumeric())]
# Filtramos para que sólo se muestren los coches que partieron desde la grilla 1 en adelante
dataset = dataset[dataset.grid.apply(lambda x: x > 0)]
# Ahora que sabemos que todos están en números, podemos convertirlo a int
dataset.position = dataset.position.astype('int')
st.write('Resultados que hayan temrinado en un número de posición y que partan de grilla 1 en adelante')
st.write(dataset)

st.subheader('Datos limpios:')
# Ahora podemos corroborar que tenemos datos completamente limpios.
st.write(dataset[["grid", "driverStanding", "constructorStanding", "position"]].agg(['min', 'max']))