from os.path import join
from typing import Tuple
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
from pyspark.sql.functions import col, year, month, dayofmonth, hour
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import plotly
import plotly.tools as tls


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


def plot_seasonal_decompose(
    result: DecomposeResult,
    dates: pd.Series = None,
    title: str = "Seasonal Decomposition",
):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name="Observed"),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name="Trend"),
            row=1,
            col=2,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name="Seasonal"),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="markers", name="Residual"),
            row=2,
            col=2,
        )
        .update_layout(
            title=f"<b>{title}</b>",
            showlegend=False,
        )
    )


@st.cache_resource
def createSARIMAModel(data: pd.DataFrame) -> SARIMAX:
    model = pm.auto_arima(
        data,
        test="adf",
        information_criterion="aic",
        trace=True,
        m=12,
        D=1,
        d=1,
        error_action="ignore",
    )
    parameters = model.get_params()

    model_fit = SARIMAX(
        data,
        order=parameters["order"],
        seasonal_order=parameters["seasonal_order"],
    ).fit(disp=-1)
    return model_fit


def createPredictedPlot(model, months: int = 12 * 4):
    yPred = model.forecast(months)
    fig = go.Figure()

    fig.add_traces(
        go.Scatter(
            x=crimesPerMonth.index,
            y=crimesPerMonth["count"],
            name="Observed",
            hovertemplate="Month: %{x}<br>" "Count: %{y}<br>" "<extra></extra>",
        )
    )

    fig.add_traces(
        go.Scatter(
            x=yPred.index,
            y=yPred,
            mode="lines",
            name="Predicted",
            hovertemplate="Month: %{x}<br>" "Count: %{y}<br>" "<extra></extra>",
        )
    )
    fig.update_layout(
        title="Monthly Crime Rate with SARIMA",
        xaxis_title="Month",
        yaxis_title="Count",
    )
    return fig


spark, _ = initializeSpark()
processedSDF = spark.read.load(
    path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet"
)

crimesPerMonth = getCrimesPerMonth(processedSDF)

multiplicative_decomposition = seasonal_decompose(
    crimesPerMonth["count"], model="multiplicative", extrapolate_trend="freq"
)

fig = plot_seasonal_decompose(multiplicative_decomposition, dates=crimesPerMonth.index)
model_fit = createSARIMAModel(crimesPerMonth["count"])


initialPredictionsPage, SARIMAModelPage, diagnosticsPage = st.tabs(
    ["Initial Predictions", "SARIMA Model", "Diagnostics"]
)


with initialPredictionsPage:
    st.title("NYC Crime Predictor Model")
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with SARIMAModelPage:
    st.plotly_chart(createPredictedPlot(model=model_fit), use_container_width=True)

with diagnosticsPage:
    diagonosticFigure = model_fit.plot_diagnostics(figsize=(15, 12))
    st.pyplot(diagonosticFigure)
