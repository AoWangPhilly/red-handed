from os.path import join
from pyspark.sql.functions import col, year, month
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from src.plot import plotCrimesVsTemp
from src.weather import read_weather_data
from src.util import initializeSpark
from src.crime import getCrimesPerMonth, getCorrelationPerCrimes
import streamlit as st

spark, _ = initializeSpark()
processedSDF = spark.read.load(
    path=join("data", "NYPD_Complaint_Data_Historic.parquet"), format="parquet"
)

crimesPerMonth = getCrimesPerMonth(processedSDF)
outsideCrimes = (
    processedSDF.select(
        "LOC_OF_OCCUR_DESC",
        year("CMPLNT_FR").alias("CMPLNT_FR_YEAR"),
        month("CMPLNT_FR").alias("CMPLNT_FR_MONTH"),
    )
    .filter((col("LOC_OF_OCCUR_DESC") != "INSIDE"))
    .cache()
)

insideCrimes = processedSDF.select(
    "LOC_OF_OCCUR_DESC",
    year("CMPLNT_FR").alias("CMPLNT_FR_YEAR"),
    month("CMPLNT_FR").alias("CMPLNT_FR_MONTH"),
).filter((col("LOC_OF_OCCUR_DESC") == "INSIDE"))

outsideCrimesDF = (
    outsideCrimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
    .count()
    .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
    .toPandas()
)

insideCrimesDF = (
    insideCrimes.groupBy(["CMPLNT_FR_YEAR", "CMPLNT_FR_MONTH"])
    .count()
    .sort([col("CMPLNT_FR_YEAR"), col("CMPLNT_FR_MONTH")])
    .toPandas()
)

typesOfCrimes = (
    processedSDF.filter((col("LOC_OF_OCCUR_DESC") != "INSIDE"))
    .select("OFNS_DESC")
    .distinct()
    .toPandas()
)

df = read_weather_data()

st.title("Temperature Correlation with Crime")

correlationsPerCrime = getCorrelationPerCrimes(
    weatherDF=df, _crimeDF=outsideCrimes, typesOfCrimes=typesOfCrimes
)

gb = GridOptionsBuilder.from_dataframe(correlationsPerCrime)
gb.configure_pagination(
    paginationAutoPageSize=True, paginationPageSize=5
)  # Add pagination
gb.configure_side_bar()  # Add a sidebar
gb.configure_selection()
gridOptions = gb.build()

data_grid, graph = st.columns(2)

with st.container():
    grid_response = AgGrid(
        correlationsPerCrime,
        gridOptions=gridOptions,
        data_return_mode="AS_INPUT",
        update_mode="MODEL_CHANGED",
        fit_columns_on_grid_load=True,
    )
    if selectRows := grid_response["selected_rows"]:
        crime = selectRows[0]["Crime"]
    else:
        crime = "ASSAULT & RELATED OFFENSES"

    st.plotly_chart(plotCrimesVsTemp(crime, df, outsideCrimesDF))
