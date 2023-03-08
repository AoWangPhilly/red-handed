from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose


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


def plot_seasonal_decompose(
        result: DecomposeResult,
        dates: pd.Series = None,
        title: str = "Seasonal Decomposition"
):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        ).add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name="Observed"),
            row=1,
            col=1,
        ).add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name="Trend"),
            row=2,
            col=1,
        ).add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name="Seasonal"),
            row=3,
            col=1,
        ).add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="markers", name="Residual"),
            row=4,
            col=1,
        ).update_layout(title=f"<b>{title}</b>", showlegend=False, height=800)
    )


spark, _ = initializeSpark()
processedSDF = spark.read.load(path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet")

crimesPerMonth = getCrimesPerMonth(processedSDF)

multiplicative_decomposition = seasonal_decompose(
    crimesPerMonth["count"],
    model="multiplicative",
    extrapolate_trend="freq"
)

fig = plot_seasonal_decompose(multiplicative_decomposition, dates=crimesPerMonth.index)

with st.container():
    st.title("Red-Handed :oncoming_police_car: :cop:")
    st.plotly_chart(fig, use_container_width=True)
