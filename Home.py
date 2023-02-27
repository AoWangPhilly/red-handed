import streamlit as st
import pandas as pd
from os.path import join
from typing import Dict
import plotly.express as px

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


@st.cache_data
def get_crime_by_year(year: int) -> pd.DataFrame:
    df = pd.read_parquet(
        join("data", f"NYPD_Complaint_Data_Historic_{year}.parquet"),
        engine="fastparquet"
    )
    return rename_values(df)


def get_all_crimes() -> Dict[int, pd.DataFrame]:
    output = {year: get_crime_by_year(year) for year in range(START_YEAR, END_YEAR + 1)}
    return output


# crimes = get_all_crimes()


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
                      "<br><i>Occurences</i>: %{y}<br>"
                      "<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


st.write("# RedHanded :wave:")

year = st.selectbox(
    label="Select a year to analyze:",
    options=range(START_YEAR, END_YEAR+1),
    index=END_YEAR-START_YEAR
)

data = get_crime_by_year(year=year)
fig = get_top_n_crimes(data)

st.plotly_chart(fig, use_container_width=True)

# st.write(crime_2021)
# crime_2021.rename(columns={"Latitude": "LAT", "Longitude": "LON"}, inplace=True)
# st.map(crime_2021)
