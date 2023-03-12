from typing import List, Tuple
from concurrent import futures

import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame

from .crime import getSpecificCrimes


@st.cache_data
def getCorrelation(
    _sdf: PySparkDataFrame, weatherData: pd.DataFrame, crime: str
) -> Tuple[str, float]:
    crimeSpecificDF = getSpecificCrimes(_sdf=_sdf, crime=crime)

    # check if the length of the crimeSpecificDF and weatherData are the same,
    # will need this for the merge
    if len(crimeSpecificDF) == len(weatherData):
        merged = crimeSpecificDF.merge(weatherData, left_index=True, right_index=True)

        # create the model and fit it
        ols = smf.ols(formula="TAVG ~ count", data=merged)
        model = ols.fit()

        # append the crime and the correlation to the output list
        return crime, model.rsquared
    return crime, -1


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
    # rename the date column and set it as the index
    weatherData = weatherDF.rename(columns={"DATE": "Date"}).set_index("Date")

    output = map(
        lambda x: getCorrelation(_sdf=_crimeDF, weatherData=weatherData, crime=x),
        typesOfCrimes["OFNS_DESC"],
    )
    output = list(filter(lambda x: x[1] != -1, output))
    # create a DataFrame from the output list and sort it by the correlation (descending)

    df = pd.DataFrame(
        sorted(output, key=lambda x: x[1], reverse=True),
        columns=["Crime", "Correlation"],
    )
    return df


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


def calculateError(model, test) -> Tuple[float, float]:
    predictions = model.forecast(len(test))
    predictions = pd.Series(predictions, index=test.index)
    residuals = test["count"] - predictions
    mape = np.mean(abs(residuals / test["count"])) * 100
    rmse = np.sqrt(np.mean(residuals**2))
    return mape, rmse
