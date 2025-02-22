import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

with open("app/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("Glossary of Modelling Terms")

with col2:
    st.image("app/assets/daa-logo.svg", width=300)

st.dataframe(
    pd.read_csv("app/glossary.csv"),
    use_container_width=True,
    hide_index=True,
)
