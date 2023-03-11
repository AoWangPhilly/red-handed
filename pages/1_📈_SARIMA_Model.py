"""SARIMA Model"""

from os.path import join

import streamlit as st
from pyspark.sql.functions import year
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import numpy as np
from statsmodels.tsa.stattools import adfuller

from src.crime import getCrimesPerMonth

from src.model import (
    createLinearRegressionModel,
    getLinearRegressionMetrics,
    calculateError,
)
from src.util import initializeSpark
from src.plot import (
    plot_seasonal_decompose,
    createPredictedPlot,
    createSARIMAModel,
    createPredictedPlot,
)

st.set_page_config(layout="wide")

# --------------------- Setup --------------------- #
# Load Spark and data
spark, _ = initializeSpark()
processedSDF = spark.read.load(
    path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet"
)

# Get crimes per month
crimesPerMonth = getCrimesPerMonth(processedSDF)

# Decompose the time series
multiplicative_decomposition = seasonal_decompose(
    crimesPerMonth["count"], model="multiplicative", extrapolate_trend="freq"
)

# Use 80% of the data for training
train_size = int(len(crimesPerMonth) * 0.8)
train = crimesPerMonth.iloc[:train_size]
test = crimesPerMonth.iloc[train_size:]

# Create the SARIMA model
model_fit = createSARIMAModel(train)
# --------------------- Setup --------------------- #


# --------------------- Create the UI --------------------- #
st.title("NYC Crime Predictor Model")

(
    initialPredictionsPage,
    seasonalDecompositionPage,
    SARIMAModelPage,
    diagnosticsPage,
) = st.tabs(
    ["Initial Predictions", "Seasonal Decomposition", "SARIMA Model", "Diagnostics"]
)

with initialPredictionsPage:
    st.write("## Initial Predictions")
    regr = createLinearRegressionModel(crimeData=crimesPerMonth)
    predicted = regr.predict(
        crimesPerMonth.index.values.reshape(-1, 1).astype("float64")
    )
    residual = crimesPerMonth["count"].values - predicted

    coef, mse, r2Score = getLinearRegressionMetrics(regr=regr, crimeData=crimesPerMonth)
    linearRegressionFig = px.scatter(
        x=crimesPerMonth.index,
        y=crimesPerMonth["count"],
        trendline="ols",
        title="Crimes per Month - Linear Regression",
    )

    linearRegressionFig.update_layout(
        xaxis_title="Month",
        yaxis_title="Count",
    )

    residualFig = px.scatter(
        x=crimesPerMonth.index, y=residual, title="Crimes per Month - Residual"
    )

    residualFig.update_layout(
        xaxis_title="Month",
        yaxis_title="Residual",
    )
    with st.expander("See explanation"):
        st.write(
            "One of my questions when considering moving to NYC was how NYC "
            "crime rates have been over the years. Has it been increasing, decreasing, "
            "or stagnant? We can check the slope with a linear regression model and see if "
            "it's positive or negative. For example, below, the slope is -0.0000153, indicating "
            "that crime has decreased over the years. However, the linear regression's R2 score "
            "is relatively low: 0.31, showing that the linear regression model isn't the best for "
            "predicting crime rates."
        )

        st.write(
            "Additionally, in the residual plot below to the right, we see that there isn't "
            "any discerning pattern from a quick glance, but looking at the plot to the left, "
            "we see that it's very similar. The shape's the same, but the scatter plot on the left "
            "has a downtrend. A good residual plot would show a random scatter of points. Therefore, "
            "it further proves that there are better fits than linear regression. "
        )

    reg_col, res_col = st.columns(2)

    with reg_col:
        st.plotly_chart(linearRegressionFig, use_container_width=True)

    with res_col:
        st.plotly_chart(residualFig, use_container_width=True)

    with st.container():
        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(label="Coefficient", value=f"{coef[0]:.2e}")

        with metric2:
            st.metric(label="RMSE", value=f"{np.sqrt(mse):,.2f}")

        with metric3:
            st.metric(label="R2 Score", value=round(r2Score, 2))

with seasonalDecompositionPage:
    st.write("### Seasonal Decomposition")

    with st.expander("See explanation"):
        st.write(
            "Seasonal decomposition breaks down a time series data into three parts: trend, "
            "seasonal, and residual. The main point of seasonal decomposition "
            "is to better understand the underlying patterns and structure of "
            "the time series data. The trend component represents the long-term changes the data, "
            "such as overall growth or decline over time. The seasonal component represents the "
            "repeating patterns in the data that occur within a year or any cadence. The residual component represents "
            "the remaining variation in the data that cannot be attributed to the trend or seasonal components. "
            "In this case we have a multiplicative decomposition:"
        )
        st.latex("y(t) = T(t) * S(t) * e(t)")

        st.write(
            "So below we see four plots: the original data, the trend, the seasonal, and the residual."
            "The observed plot shows monthly crime rates from 2006 to 2020. The trend plot shows that "
            "the trend is decreasing over time, which confirms the linear regression plot in the first tab."
            "The seasonal plot shows that there are spikes in crime in August and record low crimes in February. "
            "As for the residual plot, it shows that there is no discerning pattern, so it shows that "
            "it's a multiplicative model and not an additive model."
        )

    seasonalDecomposeFig = plot_seasonal_decompose(
        multiplicative_decomposition, dates=crimesPerMonth.index
    )

    st.plotly_chart(
        seasonalDecomposeFig,
        use_container_width=True,
    )

with SARIMAModelPage:
    st.write("### SARIMA Model")

    with st.expander("See explanation"):
        st.write("### Stationarity")
        st.write(
            "SARIMA requires stationary data because it is built around the assumption "
            "of stationarity, and the statistical properties of the data need to be constant "
            "over time for the model to be accurate and effective. Three components of stationarity would be: "
            "constant mean, constant variance, and constant autocorrelation (correlation between the values of"
            "a time series at different time points). To check for stationarity, we can use the Augmented Dickey-Fuller test. "
        )

        st.write("### Augmented Dickey-Fuller Test")
        st.write(
            "The null hypothesis of the ADF test is that the time series is non-stationary. "
            "So, if the p-value of the test is less than the significance level (0.05) then you "
            "reject the null hypothesis and infer that the time series is stationary. "
            "Using the Augmented Dickey Fuller test from the `statsmodels` package. "
            "The p-value is 0.74, which is greater than 0.05, so we cannot reject the null hypothesis. "
        )
        st.write(
            "So, in our case, if p-value > 0.05 we go ahead with finding the order of differencing. "
            "After the first-order of differencing, we see the p-value becomes a lot smaller and is "
            "ready to be used in the SARIMA model. In the model we'll have to tell the model do the "
            "first-order of differencing as a parameter. "
        )
        adf = adfuller(crimesPerMonth["count"].dropna().values, autolag="AIC")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Initial p-value", value=f"{adf[1]:.2f}")

        adf = adfuller(crimesPerMonth["count"].diff().dropna().values, autolag="AIC")
        with col2:
            st.metric(label="After differencing p-value", value=f"{adf[1]:.2e}")

    year = st.selectbox(
        "Select number of years to predict: ",
        range(1, 11),
    )
    months = year * 12
    st.plotly_chart(
        createPredictedPlot(
            model=model_fit, crimesPerMonth=crimesPerMonth, months=months
        ),
        use_container_width=True,
    )

    st.write("### Error")
    st.write(
        "MAPE measures the average absolute percentage difference between the predicted "
        "values and the actual values. It shows the average percentage difference between "
        "the predicted and actual values. In our case 6.3%, which is relatively low, meaning "
        "our model is accurate. At least much more accurate than the linear regression."
    )

    st.write(
        "RMSE measures the square root of the average of the squared differences between the "
        "predicted values and the actual values. It shows the magnitude of the errors in your "
        "predictions, and it gives more weight to large errors compared to smaller errors."
    )
    col1, col2 = st.columns(2)

    mape, rsme = calculateError(model=model_fit, test=test)

    with col1:
        st.metric(label="MAPE", value=f"{mape:.2f}%")

    with col2:
        st.metric(label="RMSE", value=f"{rsme:.2f}")

with diagnosticsPage:
    st.write("### Diagnostics")
    st.write(
        "Showing the diagnostics ensures that the residuals of the model is uncorrelated "
        "and normally distributed with mean 0. If the SARIMA model doesn't satisfy those "
        "requirements, then it needs further improvement. But we do see that below. The standardized "
        "residual on the top left shows the mean around 0 and shows no obvious signs of seasonality. "
        "Additionally, the top right plot with the orange KDE follows closely the normal distribution "
        "N(0, 1). As well at the QQ-plot followin the linear trend of the samples taken "
        "from the normal distribution."
    )
    diagonosticFigure = model_fit.plot_diagnostics(figsize=(15, 12))
    st.pyplot(diagonosticFigure, bbox_inches="tight")


# --------------------- Create the UI --------------------- #
