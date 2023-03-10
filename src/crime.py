from pyspark.sql.functions import col, year, month, dayofmonth, hour
import streamlit as st
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
import pandas as pd
from typing import List
import statsmodels.formula.api as smf


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


@st.cache_data
def getSpecificCrimes(_sdf: PySparkDataFrame, crime: str) -> pd.DataFrame:
    crimeSpecific = (
        _sdf.filter(col("OFNS_DESC") == crime)
        .groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
        .count()
        .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
        .toPandas()
    )
    crimeSpecific["Date"] = (
        crimeSpecific.CMPLNT_FR_MONTH.map(str)
        + "/"
        + crimeSpecific.CMPLNT_FR_YEAR.map(str)
    )
    crimeSpecific["Date"] = pd.to_datetime(crimeSpecific["Date"])
    crimeSpecific.set_index("Date", inplace=True)
    crimeSpecific.drop(columns=["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"], inplace=True)
    return crimeSpecific


@st.cache_data
def getCorrelationPerCrimes(
    _crimeDF, weatherDF: pd.DataFrame, typesOfCrimes: List[str]
) -> pd.DataFrame:
    output = []

    weatherData = weatherDF.rename(columns={"DATE": "Date"}).set_index("Date")
    for crime in typesOfCrimes["OFNS_DESC"]:
        print(crime)
        crimeSpecificDF = getSpecificCrimes(_sdf=_crimeDF, crime=crime)
        if len(crimeSpecificDF) == len(weatherData):
            merged = crimeSpecificDF.merge(
                weatherData, left_index=True, right_index=True
            )
            ols = smf.ols(formula="TAVG ~ count", data=merged)
            model = ols.fit()
            output.append((crime, model.rsquared))

    return pd.DataFrame(
        sorted(output, key=lambda x: x[1], reverse=True),
        columns=["Crime", "Correlation"],
    )
