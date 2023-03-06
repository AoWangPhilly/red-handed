import streamlit as st
import pandas as pd
from os.path import join
from typing import Dict
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
import numpy as np
import leafmap.foliumap as leafmap
import streamlit.components.v1 as components

# http://michael-harmon.com/blog/crimetime.html
st.set_page_config(layout="wide")

START_YEAR = 2006
END_YEAR = 2021


def rename_values(df: pd.DataFrame) -> pd.DataFrame:
    df.replace({
        'HARRASSMENT 2': 'HARASSMENT',
        'ESCAPE 3': 'ESCAPE',
        'ASSAULT 3 & RELATED OFFENSES': 'ASSAULT & RELATED OFFENSES',
        'CRIMINAL MISCHIEF & RELATED OF': 'CRIMINAL MISCHIEF',
        'OFF. AGNST PUB ORD SENSBLTY &': 'OFFENSES AGAINST PUBLIC ORDER/ADMINISTRATION',
        'OTHER STATE LAWS (NON PENAL LA': 'OTHER STATE LAWS (NON PENAL LAW)',
        'ENDAN WELFARE INCOMP': 'ENDANGERING WELFARE OF INCOMPETENT',
        'AGRICULTURE & MRKTS LAW-UNCLASSIFIED': 'AGRICULTURE & MARKETS LAW',
        'DISRUPTION OF A RELIGIOUS SERV': 'DISRUPTION OF A RELIGIOUS SERVICE',
        'LOITERING/GAMBLING (CARDS, DIC': 'GAMBLING',
        'OFFENSES AGAINST MARRIAGE UNCL': 'OFFENSES AGAINST MARRIAGE',
        'HOMICIDE-NEGLIGENT,UNCLASSIFIE': 'HOMICIDE-NEGLIGENT',
        'E': 'UNKNOWN',
        'D': 'BUSINESS/ORGANIZATION',
        'F': 'FEMALE',
        'M': 'MALE'
    }, inplace=True)
    return df


def get_crime_by_year(year: int) -> pd.DataFrame:
    df = pd.read_parquet(
        join("data", f"NYPD_Complaint_Data_Historic_{year}.parquet"),
        engine="fastparquet"
    ).iloc[:10_000]
    return rename_values(df)


def get_all_crimes() -> Dict[int, pd.DataFrame]:
    output = {year: get_crime_by_year(year) for year in range(START_YEAR, END_YEAR + 1)}
    return output


crimes = get_all_crimes()


def get_top_n_crimes(df: pd.DataFrame, nth: int = 10):
    data = df.OFNS_DESC.value_counts().iloc[:nth].sort_values(ascending=False)
    fig = px.bar(data)
    fig.update_layout(
        title=f"Top {nth} Crimes in {df.CMPLNT_FR.iloc[0].year}",
        xaxis_title="Crime Type",
        yaxis_title="Number of Occurrences",
    )
    fig.update_traces(
        hovertemplate="<i>Crime</i>: %{x}"
                      "<br><i>Occurrences</i>: %{y}<br>"
                      "<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def show_number_of_crimes_per_year(crimes: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    crimes_per_year = pd.DataFrame(pd.Series({year: len(df) for year, df in crimes.items()}, name="Count"))
    fig = go.Figure()

    line = go.Scatter(
        x=crimes_per_year.index,
        y=crimes_per_year.Count,
        hovertemplate="<i>Year</i>: %{x}"
                      "<br><i>Total Crime</i>: %{y}<br>"
                      "<extra></extra>",
        mode='lines',
        name="Actual"
    )
    fig.add_trace(line)
    fig.update_layout(
        title="Total Crime Count per Year",
        xaxis_title="Year",
        yaxis_title="Count",
    )
    fig.update_layout(showlegend=False)

    X = sm.add_constant(crimes_per_year.index)
    model = sm.OLS(crimes_per_year.Count, X).fit()
    intercept, slope = model.params

    future_x = np.array(range(START_YEAR, END_YEAR + 2))
    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=intercept + slope * future_x,
            hovertemplate="<i>Year</i>: %{x}"
                          "<br><i>Predicted Total Crime</i>: %{y}<br>"
                          "<extra></extra>",
            name="Predicted"

        )
    )
    fig.update_layout(legend_title_text='Trend', showlegend=True)
    return fig


st.write("# RedHanded :wave:")

with st.container():
    number_of_crimes_per_year_line = show_number_of_crimes_per_year(crimes=crimes)

    st.plotly_chart(number_of_crimes_per_year_line, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        year = st.selectbox(
            label="Select a year to analyze:",
            options=range(START_YEAR, END_YEAR + 1),
            index=END_YEAR - START_YEAR
        )
        crime_year = get_crime_by_year(year=year)
        top_crimes = get_top_n_crimes(crime_year)

        st.plotly_chart(top_crimes, use_container_width=True)

    with col2:
        st.plotly_chart(px.pie(crime_year, names="LAW_CAT_CD", title="Level of Offense"), use_container_width=True)

# https://huggingface.co/spaces/giswqs/Streamlit
with st.echo():
    m = leafmap.Map(center=[40.730610, -73.935242], zoom=10)
    m.add_geojson("Borough Boundaries.geojson", layer_name='NY Bouroughs')
    print(crime_year[["Latitude", "Longitude"]])
    m.add_points_from_xy(
        crime_year,
        x="Latitude",
        y="Longitude",
    )
    m.to_streamlit()

# https://data.cityofnewyork.us/Public-Safety/Police-Precincts/78dh-3ptz
# st.write(crime_2021)
# crime_2021.rename(columns={"Latitude": "LAT", "Longitude": "LON"}, inplace=True)
# st.map(crime_2021)
