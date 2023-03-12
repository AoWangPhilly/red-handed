from os.path import join
from typing import Tuple
from pyspark.sql.functions import col, year, month
from st_aggrid import GridOptionsBuilder, AgGrid
import streamlit as st
import plotly.express as px
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
import pandas as pd

from src.plot import compareCrimeRateAndTemperature, plotCrimesVsTemp
from src.weather import read_weather_data
from src.util import initializeSpark
from src.crime import getCrimesPerMonth, splitCrimeToInsideOutside, getTypesOfCrimes
from src.model import getCorrelationPerCrimes

st.set_page_config(layout="wide")


# ---------------- Setup ----------------
def draw_aggrid_df(df) -> AgGrid:
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_grid_options(domLayout="normal")
    gb.configure_selection()
    gridOptions = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        height=300,
        width="100%",
        data_return_mode="AS_INPUT",
        fit_columns_on_grid_load=True,
        update_mode="SELECTION_CHANGED",
    )

    return grid_response


# ---------------- Setup ----------------


# ---------------- Design UI ----------------


def main():
    spark, _ = initializeSpark()
    processedSDF = spark.read.load(
        path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet"
    )

    crimesPerMonth = getCrimesPerMonth(processedSDF)

    df = read_weather_data()

    (
        outsideCrimes,
        insideCrimes,
        outsideCrimesDF,
        insideCrimesDF,
    ) = splitCrimeToInsideOutside(processedSDF)

    typesOfCrimes = getTypesOfCrimes(processedSDF)

    correlationsPerCrime = getCorrelationPerCrimes(
        weatherDF=df, _crimeDF=outsideCrimes, typesOfCrimes=typesOfCrimes
    )
    st.title("Temperature Correlation with Crime")

    (discoveryPage, outsideVsInsidePage, temperatureAndCrimePage) = st.tabs(
        ["Discovery", "Outside vs. Inside", "Temperature and Crime"]
    )

    with discoveryPage:
        crimeRateAndTemperatureFig = compareCrimeRateAndTemperature(
            weatherData=df, crimeData=crimesPerMonth
        )
        st.plotly_chart(crimeRateAndTemperatureFig, use_container_width=True)

    with outsideVsInsidePage:
        insideFig = px.scatter(
            x=df.TAVG,
            y=insideCrimesDF["count"],
            trendline="ols",
            title="Indoor Crimes per Month vs. Temperature",
        )

        insideFig.update_layout(
            xaxis_title="Temperature (°C)",
            yaxis_title="Crime Rate",
        )

        outsideFig = px.scatter(
            x=df.TAVG,
            y=outsideCrimesDF["count"],
            trendline="ols",
            title="Outdoor Crimes per Month vs. Temperature",
        )

        outsideFig.update_layout(
            xaxis_title="Temperature (°C)",
            yaxis_title="Crime Rate",
        )

        st.plotly_chart(insideFig, use_container_width=True)
        st.plotly_chart(outsideFig, use_container_width=True)

    with temperatureAndCrimePage:
        grid_response = draw_aggrid_df(correlationsPerCrime)
        if selectRows := grid_response["selected_rows"]:
            crime = selectRows[0]["Crime"]
        else:
            crime = "ASSAULT & RELATED OFFENSES"
        print(crime)
        st.plotly_chart(
            plotCrimesVsTemp(crime, df, outsideCrimesDF), use_container_width=True
        )


# ---------------- Design UI ----------------

if __name__ == "__main__":
    main()
