from typing import Tuple
import datetime as dt

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


def getBoroughData(
    sdf: PySparkDataFrame, borough: str = "BROOKLYN", choosenYear: int = 2021
) -> pd.DataFrame:
    boroughColumns = [
        "BORO_NM",
        "CMPLNT_FR",
        "CMPLNT_TO",
        "RPT_DT",
        "OFNS_DESC",
        "PD_DESC",
        "LAW_CAT_CD",
    ]
    boroughCrimes = (
        sdf.select(boroughColumns)
        .filter((year("CMPLNT_FR") == choosenYear) & (col("BORO_NM") == borough))
        .cache()
        .toPandas()
    )

    boroughCrimes.CMPLNT_FR = pd.to_datetime(boroughCrimes.CMPLNT_FR)
    boroughCrimes.CMPLNT_TO = pd.to_datetime(boroughCrimes.CMPLNT_TO)
    boroughCrimes.RPT_DT = pd.to_datetime(boroughCrimes.RPT_DT) + dt.timedelta(
        hours=23, minutes=59, seconds=59
    )
    boroughCrimes["CRIME_DURATION"] = boroughCrimes.CMPLNT_TO - boroughCrimes.CMPLNT_FR
    boroughCrimes["REPORT_LATENCY"] = boroughCrimes.RPT_DT - boroughCrimes.CMPLNT_FR
    return boroughCrimes


def getPrecinctData(
    sdf: PySparkDataFrame, precinct: int = 75, choosenYear: int = 2021
) -> pd.DataFrame:
    precinctColumns = [
        "ADDR_PCT_CD",
        "CMPLNT_FR",
        "CMPLNT_TO",
        "RPT_DT",
        "OFNS_DESC",
        "PD_DESC",
        "LAW_CAT_CD",
    ]
    precinctCrimes = (
        sdf.select(precinctColumns)
        .filter((year("CMPLNT_FR") == choosenYear) & (col("ADDR_PCT_CD") == precinct))
        .cache()
        .toPandas()
    )

    precinctCrimes.CMPLNT_FR = pd.to_datetime(precinctCrimes.CMPLNT_FR)
    precinctCrimes.CMPLNT_TO = pd.to_datetime(precinctCrimes.CMPLNT_TO)
    precinctCrimes.RPT_DT = pd.to_datetime(precinctCrimes.RPT_DT) + dt.timedelta(
        hours=23, minutes=59, seconds=59
    )
    precinctCrimes["CRIME_DURATION"] = (
        precinctCrimes.CMPLNT_TO - precinctCrimes.CMPLNT_FR
    )
    precinctCrimes["REPORT_LATENCY"] = precinctCrimes.RPT_DT - precinctCrimes.CMPLNT_FR
    return precinctCrimes


@st.cache_resource
def splitCrimeToInsideOutside(
    _sdf: PySparkDataFrame,
) -> Tuple[PySparkDataFrame, PySparkDataFrame, pd.DataFrame, pd.DataFrame]:
    outsideCrimes = (
        _sdf.select(
            "LOC_OF_OCCUR_DESC",
            year("CMPLNT_FR").alias("CMPLNT_FR_YEAR"),
            month("CMPLNT_FR").alias("CMPLNT_FR_MONTH"),
        )
        .filter((col("LOC_OF_OCCUR_DESC") != "INSIDE"))
        .cache()
    )

    insideCrimes = _sdf.select(
        "LOC_OF_OCCUR_DESC",
        year("CMPLNT_FR").alias("CMPLNT_FR_YEAR"),
        month("CMPLNT_FR").alias("CMPLNT_FR_MONTH"),
    ).filter((col("LOC_OF_OCCUR_DESC") == "INSIDE"))

    outsideCrimesDF = (
        outsideCrimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
        .count()
        .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
        .toPandas()
    )

    insideCrimesDF = (
        insideCrimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
        .count()
        .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
        .toPandas()
    )
    return outsideCrimes, insideCrimes, outsideCrimesDF, insideCrimesDF


@st.cache_data
def getTypesOfCrimes(_sdf: PySparkDataFrame) -> pd.DataFrame:
    typesOfCrimes = (
        _sdf.filter((col("LOC_OF_OCCUR_DESC") != "INSIDE"))
        .select("OFNS_DESC")
        .distinct()
        .toPandas()
    )
    return typesOfCrimes
