from typing import Tuple
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import streamlit as st


@st.cache_resource
def initializeSpark() -> Tuple[SparkSession, SparkContext]:
    """Create a Spark Session for Streamlit app"""
    conf = SparkConf().setAppName("crime-processor").setMaster("local")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark, spark.sparkContext
