from os.path import join
from typing import List
from typing import Tuple

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
from pyspark.sql.functions import (
    col,
    year,
    month,
    dayofmonth,
    hour
)
from streamlit_folium import st_folium

from Home import processedSDF


@st.cache_resource
def initializeSpark() -> Tuple[SparkSession, SparkContext]:
    """Create a Spark Session for Streamlit app"""
    conf = SparkConf().setAppName("crime-processor").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext


@st.cache_data
def getCrimesPerMonth(_sdf: PySparkDataFrame) -> pd.DataFrame:
    crimeTimes = _sdf.select(
        year("CMPLNT_FR").alias("CMPLNT_FR_YEAR"),
        month("CMPLNT_FR").alias("CMPLNT_FR_MONTH"),
        dayofmonth("CMPLNT_FR").alias("CMPLNT_FR_DAY"),
        hour("CMPLNT_FR").alias("CMPLNT_FR_HOUR")
    ).cache()

    crimesPerMonth = crimeTimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"]).count().sort(
        [col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")]
    ).toPandas()

    crimesPerMonth["Date"] = crimesPerMonth.CMPLNT_FR_MONTH.map(str) + "/" + crimesPerMonth.CMPLNT_FR_YEAR.map(str)
    crimesPerMonth.Date = pd.to_datetime(crimesPerMonth.Date)
    crimesPerMonth.set_index("Date", inplace=True)
    crimesPerMonth.drop(columns=["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"], inplace=True)
    return crimesPerMonth


def createNYCMap() -> folium.folium.Map:
    # Central lat/long values of NYC
    nycCoordinates = [40.72, -73.9999]
    zoom = 10

    # instatiate a folium map object with the above coordinate at center
    nycCrimeMap = folium.Map(
        location=nycCoordinates,
        zoom_start=zoom
    )
    return nycCrimeMap


def createChloropleth(
        map_: folium.folium.Map,
        geoData: gpd.geodataframe.GeoDataFrame,
        columns: List[str],
        keyOn: str,
) -> folium.folium.Map:
    style_function = lambda x: {
        "fillColor": "#ffffff",
        "color": "#000000",
        "fillOpacity": 0.1,
        "weight": 0.1
    }

    highlight_function = lambda x: {
        "fillColor": "#000000",
        "color": "#000000",
        "fillOpacity": 0.50,
        "weight": 0.1
    }
    folium.Choropleth(
        geo_data=geoData,
        data=geoData,
        columns=columns,
        key_on=keyOn,
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=.1,
    ).add_to(map_)

    folium.features.GeoJson(
        geoData,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=columns,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(map_)

    return map_


def createBoroughMap(_sdf: PySparkDataFrame, choosenYear: int) -> folium.folium.Map:
    map_ = createNYCMap()

    boroughCrimeCountPerYear = _sdf.select("BORO_NM", year("CMPLNT_FR").alias("year")).groupby(
        ["BORO_NM", "year"]).count().cache().toPandas()

    yearCrimeCount = boroughCrimeCountPerYear[boroughCrimeCountPerYear.year == choosenYear].drop(columns=["year"])
    yearCrimeCount.BORO_NM = yearCrimeCount.BORO_NM.str.title()

    boroughGeoJson = gpd.read_file(join("geojson", "Borough Boundaries.geojson"))
    boroughDF = boroughGeoJson.merge(yearCrimeCount, left_on="boro_name", right_on="BORO_NM", how="left")
    return createChloropleth(
        map_=map_,
        geoData=boroughDF,
        columns=["boro_name", "count"],
        keyOn="feature.properties.boro_name"
    )


def createPrecinctMap(sdf: PySparkDataFrame, choosenYear: int) -> folium.folium.Map:
    map_ = createNYCMap()

    precinctCrimeCountPerYear = sdf.select("ADDR_PCT_CD", year("CMPLNT_FR").alias("year")).groupby(
        ["ADDR_PCT_CD", "year"]).count().cache().toPandas()

    yearCrimeCount = precinctCrimeCountPerYear[precinctCrimeCountPerYear.year == choosenYear].rename(
        columns={"ADDR_PCT_CD": "precinct"}).drop(columns=["year"])

    precinctGeoJson = gpd.read_file(join("geojson", "Police Precincts.geojson"))
    precinctGeoJson.precinct = precinctGeoJson.precinct.astype("Int32")
    precinctDF = precinctGeoJson.merge(yearCrimeCount, on="precinct", how="left")

    return createChloropleth(
        map_=map_,
        geoData=precinctDF,
        columns=["precinct", "count"],
        keyOn="feature.properties.precinct"
    )


st.title("Crime in Boroughs and Precincts")

col1, col2 = st.columns(2)

with st.form("map-form"):
    boroughOrPrecinct = st.selectbox("Select Type:", ("Borough", "Precinct"))
    chooseYear = st.selectbox("Select Year:", range(2006, 2022))
    submit = st.form_submit_button("Submit")

with st.container():
    if submit:

        NYCMap = createBoroughMap(processedSDF,
                                  choosenYear=chooseYear) if boroughOrPrecinct == "Borough" else createPrecinctMap(
            processedSDF, choosenYear=chooseYear)
        st_data = st_folium(NYCMap)
        st.session_state.map = NYCMap
    else:
        if map_ := st.session_state.get("map"):
            st_folium(map_)
