from os.path import join
import pandas as pd
import streamlit as st


@st.cache_data
def read_weather_data(fname: str = join("data", "USW00094728.csv")) -> pd.DataFrame:
    df = pd.read_csv(fname, parse_dates=["DATE"])
    subset_df = df[["DATE", "TAVG", "TMAX", "TMIN", "PRCP", "SNOW"]]
    timerange = subset_df.query("DATE.between('2006-01-01', '2021-12-31')")
    return timerange.reset_index(drop=True)
