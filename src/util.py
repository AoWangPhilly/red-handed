from typing import Tuple

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import streamlit as st


@st.cache_resource
def initializeSpark() -> Tuple[SparkSession, SparkContext]:
    """Initialize Spark for the Streamlit app"""
    conf = SparkConf().setAppName("crime-processor").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext


def convert_to_am_pm(hour: int) -> str:
    """Converts a 24-hour time to AM/PM

    Args:
        hour (int): The hour to convert

    Returns:
        str: The hour in AM/PM format
    """
    if hour == 0:
        return "12 AM"
    elif 1 <= hour <= 11:
        return f"{hour} AM"
    elif hour == 12:
        return "12 PM"
    else:
        return f"{hour - 12} PM"
