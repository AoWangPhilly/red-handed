from statsmodels.tsa.seasonal import DecomposeResult
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import numpy as np
import pmdarima as pm
import plotly.express as px

from .crime import getSpecificCrimes
from .util import convert_to_am_pm


def plot_seasonal_decompose(
    result: DecomposeResult,
    dates: pd.Series = None,
    title: str = "Seasonal Decomposition",
):
    """Plot the seasonal decomposition of a time series

    Args:
        result (DecomposeResult): The result of the seasonal decomposition
        dates (pd.Series, optional): The x-axis values. Defaults to None.
        title (str, optional): The title for the plot. Defaults to "Seasonal Decomposition".

    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
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
            height=800,
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
    """Create a SARIMA model

    Returns:
        SARIMAX: The SARIMA model
    """
    parameters = model.get_params()

    model_fit = SARIMAX(
        data,
        order=parameters["order"],
        seasonal_order=parameters["seasonal_order"],
    ).fit(disp=-1)
    return model_fit


def createPredictedPlot(model, crimesPerMonth: pd.DataFrame, months: int = 12 * 4):
    """Create a plot of the predicted values

    Args:
        model (SARIMAX): The SARIMA model
        crimesPerMonth (pd.DataFrame): The crime data
        months (int, optional): The number of months to predict. Defaults to 12*4, 4 years.

    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
    yPred = model.predict(start=len(crimesPerMonth), end=len(crimesPerMonth) + months)
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
    """Plot the crimes vs. temperature

    Args:
        crime (str): The crime to plot
        weatherDF (pd.DataFrame): The weather data
        outsideCrimes (pd.DataFrame): The crime data

    Returns:
        plotly.graph_objects.Figure: The plotly figure
    """
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


def compareCrimeRateAndTemperature(weatherData: pd.DataFrame, crimeData: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Temperature", "Crime Rate"),
        shared_xaxes=True,
    )

    average_temperature = go.Scatter(
        x=weatherData.DATE,
        y=weatherData.TAVG,
        hovertemplate="<i>Date</i>: %{x}"
        "<br><i>Temperature</i>: %{y}°C<br>"
        "<extra></extra>",
        mode="lines",
        name="Average Temperature",
    )

    monthlyCrime = go.Scatter(
        x=crimeData.index,
        y=crimeData["count"],
        hovertemplate="<i>Date</i>: %{x}"
        "<br><i>Count</i>: %{y}<br>"
        "<extra></extra>",
        mode="lines",
        name="Monthly Crime Rates",
    )

    fig.add_trace(average_temperature, row=1, col=1)
    fig.add_trace(monthlyCrime, row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="Month", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        title="Temperature and Crime Rates from 2006-2021",
        height=600,
    )
    return fig


def getTopNCrimePlot(df: pd.DataFrame, n: int = 10):
    try:
        title = f"Top {n} Crimes in {df.BORO_NM.iloc[0].title()}"
    except Exception:
        title = f"Top {n} Crimes in Precinct {df.ADDR_PCT_CD.iloc[0]}"

    fig = px.bar(
        df.OFNS_DESC.value_counts()[:n],
        title=title,
    )

    fig.update_layout(
        xaxis_title="Types of Crimes",
        yaxis_title="Crime Count",
        showlegend=False,
    )
    fig.update_traces(
        hovertemplate="<i>Crime: </i> %{x}\n"
        "<br><i>Count: </i> %{y}"
        "<extra></extra>"
    )
    return fig


def crimeFrequencySubplot(df: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Crimes per Month",
            "Crimes per Day of Week",
            "Crimes per Day",
            "Crimes per Hour",
        ),
    )
    weekNameMapping = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }

    df = df.copy(deep=True)
    df["MONTH"] = df.CMPLNT_FR.dt.month
    df["MONTH_NAME"] = df.CMPLNT_FR.dt.month_name()
    df["DAY"] = df.CMPLNT_FR.dt.day
    df["DAY_OF_WEEK"] = df.CMPLNT_FR.dt.day_name()
    df["HOUR"] = df.CMPLNT_FR.dt.hour
    df["COUNT"] = 1

    monthly = (
        df[["MONTH", "MONTH_NAME", "COUNT"]]
        .groupby(["MONTH", "MONTH_NAME"])
        .sum()
        .reset_index()
    )

    weekly = df.DAY_OF_WEEK.value_counts()
    weekly.index = weekly.index.map(weekNameMapping)
    weekly.sort_index(inplace=True)
    weekly.index = weekly.index.map(
        {value: key for key, value in weekNameMapping.items()}
    )

    daily = df.DAY.value_counts().sort_index()

    hourly = df.HOUR.value_counts().sort_index()
    hourly.index = hourly.index.map(convert_to_am_pm)

    fig.add_trace(
        go.Bar(x=monthly.MONTH_NAME, y=monthly.COUNT, name="Types of Crimes per Month"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=weekly.index, y=weekly.values, name="Types of Crimes per Day of Week"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=daily.index,
            y=daily.values,
            name="Types of Crimes per Day of Month",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=hourly.index, y=hourly.values, name="Types of Crimes per Hour"),
        row=2,
        col=2,
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_xaxes(title_text="Hours", row=2, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    # Update hover template
    fig.update_traces(
        hovertemplate="<i>Month: </i> %{x}\n"
        "<br><i>Count: </i> %{y}"
        "<extra></extra>",
        row=1,
        col=1,
    )
    fig.update_traces(
        hovertemplate="<i>Day of Week: </i> %{x}\n"
        "<br><i>Count: </i> %{y}"
        "<extra></extra>",
        row=1,
        col=2,
    )
    fig.update_traces(
        hovertemplate="<i>Day: </i> %{x}\n" "<br><i>Count: </i> %{y}" "<extra></extra>",
        row=2,
        col=1,
    )
    fig.update_traces(
        hovertemplate="<i>Hour: </i> %{x}\n"
        "<br><i>Count: </i> %{y}"
        "<extra></extra>",
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800,
        width=1000,
        title_text="Crimes in Different Time Granuarities",
        showlegend=False,
    )

    return fig


def crimeDurationAndReportLatency(df: pd.DataFrame):
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Crime Duration", "Report Crime Latency")
    )
    df = df.copy(deep=True)
    df["CRIME_DURATION"] /= np.timedelta64(1, "D")
    df["REPORT_LATENCY"] /= np.timedelta64(1, "D")
    df["REPORT_LATENCY"] = df["REPORT_LATENCY"][df["REPORT_LATENCY"] > 0]

    fig.add_trace(
        go.Histogram(
            x=df["CRIME_DURATION"],
            nbinsx=100,
            histnorm="probability",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df["REPORT_LATENCY"],
            nbinsx=100,
            histnorm="probability",
        ),
        row=2,
        col=1,
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Number of Days", row=1, col=1)
    fig.update_xaxes(title_text="Number of Days", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    # Update hover template
    fig.update_traces(
        hovertemplate="<i>Crime lasted: </i> %{x} days\n"
        "<br><i>Percentage: </i> %{y}"
        "<extra></extra>",
        row=1,
        col=1,
    )
    fig.update_traces(
        hovertemplate="<i>Reporting took: </i> %{x} days\n"
        "<br><i>Percentage: </i> %{y}"
        "<extra></extra>",
        row=2,
        col=1,
    )

    fig.update_yaxes(type="log")
    fig.update_layout(height=800, width=1000, showlegend=False)

    return fig
