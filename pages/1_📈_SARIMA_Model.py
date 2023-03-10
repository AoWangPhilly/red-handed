from os.path import join

import streamlit as st
from pyspark.sql.functions import year
from statsmodels.tsa.seasonal import seasonal_decompose
from src.crime import getCrimesPerMonth
from src.util import initializeSpark
from src.plot import (
    plot_seasonal_decompose,
    createPredictedPlot,
    createSARIMAModel,
    createPredictedPlot,
)

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


st.title("NYC Crime Predictor Model")

initialPredictionsPage, SARIMAModelPage, diagnosticsPage = st.tabs(
    ["Initial Predictions", "SARIMA Model", "Diagnostics"]
)

with initialPredictionsPage:
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with SARIMAModelPage:
    year = st.selectbox(
        "How far to predict: ",
        range(1, 11),
    )
    months = year * 12
    st.plotly_chart(
        createPredictedPlot(
            model=model_fit, crimesPerMonth=crimesPerMonth, months=months
        ),
        use_container_width=True,
    )

with diagnosticsPage:
    diagonosticFigure = model_fit.plot_diagnostics(figsize=(15, 12))
    st.pyplot(diagonosticFigure)
