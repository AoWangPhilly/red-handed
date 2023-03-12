from os.path import join
from typing import Tuple

import pandas as pd
import streamlit as st
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
from pyspark.sql.functions import col, year, month, dayofmonth, hour
from streamlit_folium import st_folium

from src.crime import getBoroughData, getPrecinctData
from src.plot import (
    crimeDurationAndReportLatency,
    crimeFrequencySubplot,
    getTopNCrimePlot,
)
from src.map import createBoroughMap, createPrecinctMap

st.set_page_config(layout="wide")


# ---------------- Setup ----------------
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
        hour("CMPLNT_FR").alias("CMPLNT_FR_HOUR"),
    ).cache()

    crimesPerMonth = (
        crimeTimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
        .count()
        .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
        .toPandas()
    )

    crimesPerMonth["Date"] = (
        crimesPerMonth.CMPLNT_FR_MONTH.map(str)
        + "/"
        + crimesPerMonth.CMPLNT_FR_YEAR.map(str)
    )
    crimesPerMonth.Date = pd.to_datetime(crimesPerMonth.Date)
    crimesPerMonth.set_index("Date", inplace=True)
    crimesPerMonth.drop(columns=["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"], inplace=True)
    return crimesPerMonth


def getBoroughFromMap(st_data) -> str:
    """Returns the borough from the map
    Args:
        st_data (dict): The data from the map
    Returns:
        str: The borough
    """
    if st_data.get("last_active_drawing") is None:
        return ""

    return st_data["last_active_drawing"]["properties"]["BORO_NM"]


def getPrecinctFromMap(st_data) -> str:
    """Returns the precinct from the map
    Args:
        st_data (dict): The data from the map
    Returns:
        str: The precinct
    """
    if st_data.get("last_active_drawing") is None:
        return ""

    return st_data["last_active_drawing"]["properties"]["precinct"]


# ---------------- Setup ----------------


# ---------------- Design UI ----------------
def main():
    spark, _ = initializeSpark()
    processedSDF = spark.read.load(
        path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet"
    )

    st.title("Crime in Boroughs and Precincts")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Tutorial")
        st.write(
            "Select a borough or precinct to view the crime map then select a year to view the crime map for that year."
            "Once the map loads, the areas highlighted in red are the areas with the highest crime rate."
            "The areas highlighted in yellow are the areas with the lowest crime rate. Additionally, "
            "the areas are clickable and the charts will react on which area you click on."
        )
        with st.form("map-form"):
            boroughOrPrecinct = st.selectbox("Select Type:", ("Borough", "Precinct"))
            chooseYear = st.selectbox("Select Year:", range(2006, 2022))
            submit = st.form_submit_button("Submit")
    with col2:
        if submit:
            NYCMap = (
                createBoroughMap(processedSDF, choosenYear=chooseYear)
                if boroughOrPrecinct == "Borough"
                else createPrecinctMap(processedSDF, choosenYear=chooseYear)
            )
            st_data = st_folium(NYCMap, width="100%", height="440")
            st.session_state.map = NYCMap
        else:
            if map_ := st.session_state.get("map"):
                st_data = st_folium(map_, width="100%", height="440")

                placeName = (
                    getBoroughFromMap(st_data)
                    if boroughOrPrecinct == "Borough"
                    else getPrecinctFromMap(st_data)
                )
                st.session_state.area = placeName

    if placeName := st.session_state.get("area"):
        try:
            if boroughOrPrecinct == "Borough":
                data = getBoroughData(
                    sdf=processedSDF, borough=placeName.upper(), choosenYear=chooseYear
                )
            elif boroughOrPrecinct == "Precinct":
                data = getPrecinctData(
                    sdf=processedSDF, precinct=int(placeName), choosenYear=chooseYear
                )

            topNCrimeFig = getTopNCrimePlot(data)
            frequencyFig = crimeFrequencySubplot(data)
            durationAndLatency = crimeDurationAndReportLatency(data)
            st.plotly_chart(topNCrimeFig, use_container_width=True)
            st.plotly_chart(frequencyFig, use_container_width=True)
            st.plotly_chart(durationAndLatency, use_container_width=True)
        except Exception as e:
            pass


# ---------------- Design UI ----------------

if __name__ == "__main__":
    main()
