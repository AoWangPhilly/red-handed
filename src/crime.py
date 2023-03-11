import streamlit as st
import pandas as pd
from pyspark.sql.functions import col, year, month, dayofmonth, hour
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame


@st.cache_data
def getCrimesPerMonth(_sdf: PySparkDataFrame) -> pd.DataFrame:
    """Returns a DataFrame with the number of crimes per month

    Args:
        _sdf (PySparkDataFrame): The Spark DataFrame to use

    Returns:
        pd.DataFrame: A DataFrame with the number of crimes per month
    """
    # cache the DataFrame for faster access
    crimeTimes = _sdf.select(
        year("CMPLNT_FR").alias("CMPLNT_FR_YEAR"),
        month("CMPLNT_FR").alias("CMPLNT_FR_MONTH"),
        dayofmonth("CMPLNT_FR").alias("CMPLNT_FR_DAY"),
        hour("CMPLNT_FR").alias("CMPLNT_FR_HOUR"),
    ).cache()

    # group by year and month and count the number of crimes
    crimesPerMonth = (
        crimeTimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
        .count()
        .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
        .toPandas()
    )

    # create a date column and set it as the index
    crimesPerMonth["Date"] = (
        crimesPerMonth.CMPLNT_FR_MONTH.map(str)
        + "/"
        + crimesPerMonth.CMPLNT_FR_YEAR.map(str)
    )
    crimesPerMonth.Date = pd.to_datetime(crimesPerMonth.Date)
    crimesPerMonth.set_index("Date", inplace=True)

    # drop the year and month columns
    crimesPerMonth.drop(columns=["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"], inplace=True)
    return crimesPerMonth


@st.cache_data
def getSpecificCrimes(_sdf: PySparkDataFrame, crime: str) -> pd.DataFrame:
    """Returns a DataFrame with the number of crimes per month for a specific crime

    Args:
        _sdf (PySparkDataFrame): The Spark DataFrame to use
        crime (str): The crime to get the data for

    Returns:
        pd.DataFrame: A DataFrame with the number of crimes per month for a specific crime
    """
    # get the specific crime
    crimeSpecific = (
        _sdf.filter(col("OFNS_DESC") == crime)
        .groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
        .count()
        .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
        .toPandas()
    )

    # create a date column and set it as the index
    crimeSpecific["Date"] = (
        crimeSpecific.CMPLNT_FR_MONTH.map(str)
        + "/"
        + crimeSpecific.CMPLNT_FR_YEAR.map(str)
    )
    crimeSpecific["Date"] = pd.to_datetime(crimeSpecific["Date"])
    crimeSpecific.set_index("Date", inplace=True)

    # drop the year and month columns
    crimeSpecific.drop(columns=["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"], inplace=True)
    return crimeSpecific
