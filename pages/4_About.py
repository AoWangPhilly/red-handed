import streamlit as st

st.set_page_config(
    layout="wide",
    page_icon=":city_sunset:",
)

with open("README.md", "r") as f:
    readme = f.read()

st.write("\n".join(readme.split("\n")[1:]))
