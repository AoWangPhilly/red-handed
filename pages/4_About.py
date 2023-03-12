import streamlit as st

with open("README.md", "r") as f:
    readme = f.read()

st.write("\n".join(readme.split("\n")[1:]))
