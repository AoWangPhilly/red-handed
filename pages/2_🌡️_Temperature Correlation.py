from os.path import join
from st_aggrid import GridOptionsBuilder, AgGrid
import streamlit as st
import plotly.express as px

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
        height=500,
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
        st.write("### Discovery")
        with st.expander("See explanation"):
            st.write(
                "Here we can see that both datasets have seasonality, "
                "that in the summer months (June, July, August), the temperature "
                "is higher, and crime rates also increase. And in the winter months "
                "(December, January, and February), it's colder, and crime rates decrease. "
            )

            st.write(
                "One possible explanation for this is that higher temperatures can "
                "increase aggressive behavior and irritability, which can lead to a "
                "rise in violent crime. Higher temperatures may also increase social "
                "gatherings and outdoor activities, creating more opportunities for "
                "property crimes such as theft and burglary."
            )

            st.write(
                "However, it is important to note that other factors, such as "
                "socioeconomic status, population density, and police presence, "
                "can also significantly impact crime rates. Therefore, it is difficult "
                "to say with certainty that temperature is the sole cause of changes in crime rates."
            )
        crimeRateAndTemperatureFig = compareCrimeRateAndTemperature(
            weatherData=df, crimeData=crimesPerMonth
        )
        st.plotly_chart(crimeRateAndTemperatureFig, use_container_width=True)

    with outsideVsInsidePage:
        st.write("### Outside vs. Inside")
        st.write(
            "Below we see a stark difference between the r-squared values of crimes that "
            "have occurred inside vs. outside. For crimes that have occurred inside, the "
            "temperature has no effect. The r-squared value is 0.00049. And temperatures "
            "affect crimes that have occurred outside considerably. The r-squared value is "
            "0.63 with a positive, medium-strong correlation. So as temperature increases to "
            "a point, the crime rate will also increase."
        )
        with st.expander("See explanation"):
            st.write(
                "When doing some EDA on how temperature affects crime rates, "
                "I saw that the linear regression model has a weak, negative correlation. "
                "However, I thought I could split crimes by whether they've occurred inside "
                "or outside. Temperature can directly impact people's behavior and activities "
                "outside, as they are more exposed to the elements."
            )

            st.write(
                "For example, people may be more likely to gather outside during hot "
                "temperatures, increasing the potential for conflicts and violent behavior. "
                "Higher temperatures may also lead to increased alcohol consumption, which can "
                "further exacerbate aggressive behavior."
            )

            st.write(
                "On the other hand, crimes inside buildings, such as homes or offices, "
                "may be less affected by temperature, as the indoor climate can be "
                "regulated by air conditioning or heating systems. However, it is "
                "essential to note that indoor temperature can still impact people's moods "
                "and behavior, especially if it is incredibly uncomfortable or if there is "
                "a malfunction in the heating or cooling system."
            )

        col1, col2 = st.columns(2)

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

        with col1:
            st.plotly_chart(insideFig, use_container_width=True)
        with col2:
            st.plotly_chart(outsideFig, use_container_width=True)

    with temperatureAndCrimePage:
        st.write("### Specific Crime vs. Temperature")
        st.write(
            "Below shows the correlation between temperature and a specific crime."
            "You can select a crime from the table below to see the correlation between "
            "temperature and that crime with a scatter plot and the linear regression line"
        )
        col1, col2 = st.columns(2)

        with col1:
            grid_response = draw_aggrid_df(correlationsPerCrime)

        if selectRows := grid_response["selected_rows"]:
            crime = selectRows[0]["Crime"]
        else:
            crime = "ASSAULT & RELATED OFFENSES"

        with col2:
            st.plotly_chart(
                plotCrimesVsTemp(crime, df, outsideCrimesDF), use_container_width=True
            )


# ---------------- Design UI ----------------

if __name__ == "__main__":
    main()
