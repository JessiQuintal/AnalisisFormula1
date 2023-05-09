import pandas as pd
import pymongo
import streamlit as st

import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
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