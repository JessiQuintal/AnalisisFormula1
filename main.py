import streamlit as st
import pandas as pd
import pymongo
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def pagina_inicio():
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
    return dataset

def pagina_graficas(dataset):
    st.title('Graficando los datos y sus relaciones')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Gráfica 1
    plt.figure(figsize=(12, 8))
    plt.hist(dataset["driverStanding"])
    plt.title("Histograma de Driver Standing")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Gráfica 2
    plt.figure(figsize=(12, 8))
    plt.hist(dataset["constructorStanding"])
    plt.title("Histograma de Constructor Standing")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Gráfica 3
    plt.figure(figsize=(12, 8))
    plt.hist(dataset["grid"])
    plt.title("Histograma de Grid")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Gráfica 4
    plt.figure(figsize=(12, 8))
    plt.hist(dataset["position"])
    plt.title("Histograma de Position")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write('Ahora obtendremos la relación que existe entre la posición final de los pilotos y las posiciones que tienen dentro del campeonato de pilotos y por equipos. Visto primero de manera estadísica y luego de manera visual.')
    st.write(dataset.corr()["position"])
    plt.figure(figsize=(12, 8))
    pd.plotting.scatter_matrix(dataset, figsize=(12, 8))
    plt.title("Matriz de Dispersión")
    st.pyplot()

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    max_grid = dataset.grid.max()
    max_position = dataset.position.max()
    max_d_position = dataset.driverStanding.max()
    max_c_position = dataset.constructorStanding.max()

    ax[0].hist2d(dataset.grid, dataset.position, (max_grid, max_position), cmap='plasma', cmin=1)
    ax[0].set_xlabel("Grid")
    ax[0].set_ylabel("Final position")

    ax[1].hist2d(dataset.driverStanding, dataset.position, (max_d_position, max_position), cmap='plasma', cmin=1)
    ax[1].set_xlabel("Driver standing")

    ax[2].hist2d(dataset.constructorStanding, dataset.position, (max_c_position, max_position), cmap='plasma', cmin=1)
    ax[2].set_xlabel("Constructor standing")

    st.pyplot(fig)


def pagina_modelo(dataset):
    st.image('logoF1.png', width=200)
    st.markdown('<h1 style="color:#800020;">Modelo de regresión lineal</h1>', unsafe_allow_html=True)
    st.markdown("""
        Consultar posiciones para predecir resultados del Gran Premio de Miami 2023:
        <br>
        [https://www.total-motorsport.com/f1-driver-constructor-standings-after-2023-azerbaijan-grand-prix/](https://www.total-motorsport.com/f1-driver-constructor-standings-after-2023-azerbaijan-grand-prix/)
        """, unsafe_allow_html=True)

    df = dataset

    # separar los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(df[['driverStanding', 'constructorStanding', 'grid']],
                                                        df['position'], test_size=0.3)
    # crear el modelo de regresión lineal
    model = LinearRegression()
    # entrenar el modelo con el conjunto de entrenamiento
    model.fit(X_train, y_train)
    # hacer predicciones con el conjunto de prueba
    y_pred = model.predict(X_test)
    # evaluar el rendimiento del modelo
    print(model.score(X_test, y_test))


    # Estilos del formulario
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)
    with st.form("modelo_form"):
        # Estilos de los campos de entrada
        st.markdown('<div style="margin-bottom:10px;">', unsafe_allow_html=True)
        driverStanding = st.number_input("Posición del Piloto", min_value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom:10px;">', unsafe_allow_html=True)
        constructorStanding = st.number_input("Posición del Constructor", min_value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom:10px;">', unsafe_allow_html=True)
        grid = st.number_input("Posición en la Parrilla", min_value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        # Estilos del botón de envío
        st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
        submit_button = st.form_submit_button(label='Predecir')
        st.markdown('</div>', unsafe_allow_html=True)

    if submit_button:
        # separar los datos en conjunto de entrenamiento y conjunto de prueba
        X_train, X_test, y_train, y_test = train_test_split(df[['driverStanding', 'constructorStanding', 'grid']],
                                                            df['position'], test_size=0.3)
        # crear el modelo de regresión lineal
        model = LinearRegression()

        # entrenar el modelo con el conjunto de entrenamiento
        model.fit(X_train, y_train)

        # hacer predicciones con los valores ingresados en el formulario
        x = np.array([driverStanding, constructorStanding, grid]).reshape(1, -1)
        y_pred = model.predict(x)

        # mostrar la predicción en la interfaz de Streamlit
        st.success(f"Quedará en el lugar #: {y_pred.round()}")

# Agregar botones de navegación
pagina_actual = st.sidebar.radio("Navegación", ("Inicio", "Gráficas", "Modelo"))

# Mostrar contenido según la página seleccionada
if pagina_actual == "Inicio":
    if 'dataset' not in st.session_state:
        dataset = pagina_inicio()
        st.session_state['dataset'] = dataset
    else:
        dataset = st.session_state['dataset']
elif pagina_actual == "Gráficas":
    if 'dataset' not in st.session_state:
        st.warning("Debes primero ir a la página 'Inicio' para generar el dataset.")
    else:
        dataset = st.session_state['dataset']
        pagina_graficas(dataset)
elif pagina_actual == "Modelo":
    if 'dataset' not in st.session_state:
        st.warning("Debes primero ir a la página 'Inicio' para generar el dataset.")
    else:
        dataset = st.session_state['dataset']
        pagina_modelo(dataset)