from os.path import join
from typing import Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
from pyspark.sql.functions import (
    col,
    year,
    month,
    dayofmonth,
    hour
)


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


spark, _ = initializeSpark()
processedSDF = spark.read.load(path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet")

crimesPerMonth = getCrimesPerMonth(processedSDF)

monthly = px.line(crimesPerMonth, x=crimesPerMonth.index, y="count")
monthly
