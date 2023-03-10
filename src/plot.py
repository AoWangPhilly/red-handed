from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import numpy as np
import pmdarima as pm
import plotly.express as px
from src.crime import getSpecificCrimes


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


def createPredictedPlot(model, crimesPerMonth: pd.DataFrame, months: int = 12 * 4):
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


def plotCrimesVsTemp(crime: str, weatherDF: pd.DataFrame, outsideCrimes: pd.DataFrame):
    crimeDF = getSpecificCrimes(_sdf=outsideCrimes, crime=crime)

    fig = px.scatter(
        x=weatherDF.TAVG,
        y=crimeDF["count"],
        trendline="ols",
        title=f"Crimes per Month vs. Temperature for {crime.title()}",
    )

    fig.update_layout(
        xaxis_title="Temperature (°C)",
        yaxis_title="Crime Rate",
    )
    return fig
