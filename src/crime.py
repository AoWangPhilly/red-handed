from typing import List

import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from pyspark.sql.functions import col, year, month, dayofmonth, hour
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


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


@st.cache_data
def getCorrelationPerCrimes(
    _crimeDF, weatherDF: pd.DataFrame, typesOfCrimes: List[str]
) -> pd.DataFrame:
    """Returns a DataFrame with the correlation between the number of crimes and the temperature

    Args:
        _crimeDF (_type_): The Spark DataFrame to use
        weatherDF (pd.DataFrame): The DataFrame with the weather data
        typesOfCrimes (List[str]): the list of crimes to get the correlation for

    Returns:
        pd.DataFrame: A DataFrame with the correlation between the number of crimes and the temperature
    """
    output = []

    # rename the date column and set it as the index
    weatherData = weatherDF.rename(columns={"DATE": "Date"}).set_index("Date")

    # loop through the crimes and get the correlation
    for crime in typesOfCrimes["OFNS_DESC"]:
        print(crime)
        crimeSpecificDF = getSpecificCrimes(_sdf=_crimeDF, crime=crime)

        # check if the length of the crimeSpecificDF and weatherData are the same,
        # will need this for the merge
        if len(crimeSpecificDF) == len(weatherData):
            merged = crimeSpecificDF.merge(
                weatherData, left_index=True, right_index=True
            )

            # create the model and fit it
            ols = smf.ols(formula="TAVG ~ count", data=merged)
            model = ols.fit()

            # append the crime and the correlation to the output list
            output.append((crime, model.rsquared))

    # create a DataFrame from the output list and sort it by the correlation (descending)
    return pd.DataFrame(
        sorted(output, key=lambda x: x[1], reverse=True),
        columns=["Crime", "Correlation"],
    )


def getLinearRegressionMetrics(regr, crimeData):
    predicted = regr.predict(crimeData.index.values.reshape(-1, 1).astype("float64"))
    mse = mean_squared_error(crimeData["count"], predicted)
    r2Score = r2_score(crimeData["count"], predicted)
    return regr.coef_, mse, r2Score


def createLinearRegressionModel(crimeData: pd.DataFrame):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(crimeData.index.values.reshape(-1, 1), crimeData["count"])
    return regr
