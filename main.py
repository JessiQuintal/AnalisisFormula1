import importlib
import os
import pandas as pd
import pymongo
import streamlit as st

st.set_page_config(
    page_title="Analisis Fórmula 1",
    page_icon="👋",
)

st.write("Análisis Fórmula 1 👋")

pages = [f[:-3] for f in os.listdir("AnalisisFormula1/Pages") if f.endswith(".py")]
selection = st.sidebar.radio("Pages:", pages)
page_module = importlib.import_module(f"Pages.{selection}")
page_module.show()

