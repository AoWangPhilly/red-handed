from os.path import join
from typing import List
import streamlit as st
import folium
import geopandas as gpd
from pyspark.sql.dataframe import DataFrame as PySparkDataFrame
from pyspark.sql.functions import year


def createNYCMap() -> folium.folium.Map:
    """Creates a map of NYC

    Returns:
        folium.folium.Map: A map of NYC
    """
    # Central lat/long values of NYC
    nycCoordinates = [40.72, -73.9999]
    zoom = 10

    # instatiate a folium map object with the above coordinate at center
    nycCrimeMap = folium.Map(location=nycCoordinates, zoom_start=zoom)
    return nycCrimeMap


def createChloropleth(
    map_: folium.folium.Map,
    geoData: gpd.geodataframe.GeoDataFrame,
    columns: List[str],
    keyOn: str,
) -> folium.folium.Map:
    """Creates a chloropleth map

    Args:
        map_ (folium.folium.Map): the map to add the chloropleth to
        geoData (gpd.geodataframe.GeoDataFrame): the GeoDataFrame to use
        columns (List[str]): the columns to use
        keyOn (str): the key to use

    Returns:
        folium.folium.Map: the map with the chloropleth added
    """
    style_function = lambda x: {
        "fillColor": "#ffffff",
        "color": "#000000",
        "fillOpacity": 0.1,
        "weight": 0.1,
    }

    highlight_function = lambda x: {
        "fillColor": "#000000",
        "color": "#000000",
        "fillOpacity": 0.50,
        "weight": 0.1,
    }

    # add a choropleth layer to the map
    folium.Choropleth(
        geo_data=geoData,
        data=geoData,
        columns=columns,
        key_on=keyOn,
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.1,
    ).add_to(map_)

    # add a GeoJson layer to the map
    folium.features.GeoJson(
        geoData,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=columns,
            style=(
                "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
            ),
        ),
    ).add_to(map_)

    return map_


@st.cache_resource
def createBoroughMap(_sdf: PySparkDataFrame, choosenYear: int) -> folium.folium.Map:
    """Creates a map of NYC with the boroughs colored by the number of crimes in a given year

    Args:
        _sdf (PySparkDataFrame): the Spark DataFrame to use
        choosenYear (int): the year to use

    Returns:
        folium.folium.Map: the map with the boroughs colored by the number of crimes in a given year
    """
    map_ = createNYCMap()

    boroughCrimeCountPerYear = (
        _sdf.select("BORO_NM", year("CMPLNT_FR").alias("year"))
        .groupby(["BORO_NM", "year"])
        .count()
        .cache()
        .toPandas()
    )

    yearCrimeCount = boroughCrimeCountPerYear[
        boroughCrimeCountPerYear.year == choosenYear
    ].drop(columns=["year"])
    yearCrimeCount.BORO_NM = yearCrimeCount.BORO_NM.str.title()

    boroughGeoJson = gpd.read_file(join("geojson", "Borough Boundaries.geojson"))
    boroughDF = boroughGeoJson.merge(
        yearCrimeCount, left_on="boro_name", right_on="BORO_NM", how="left"
    )
    return createChloropleth(
        map_=map_,
        geoData=boroughDF,
        columns=["boro_name", "count"],
        keyOn="feature.properties.boro_name",
    )


@st.cache_resource
def createPrecinctMap(_sdf: PySparkDataFrame, choosenYear: int) -> folium.folium.Map:
    """Creates a map of NYC with the precincts colored by the number of crimes in a given year

    Args:
        _sdf (PySparkDataFrame): the Spark DataFrame to use
        choosenYear (int): the year to use

    Returns:
        folium.folium.Map: the map with the precincts colored by the number of crimes in a given year
    """
    map_ = createNYCMap()

    precinctCrimeCountPerYear = (
        _sdf.select("ADDR_PCT_CD", year("CMPLNT_FR").alias("year"))
        .groupby(["ADDR_PCT_CD", "year"])
        .count()
        .cache()
        .toPandas()
    )

    yearCrimeCount = (
        precinctCrimeCountPerYear[precinctCrimeCountPerYear.year == choosenYear]
        .rename(columns={"ADDR_PCT_CD": "precinct"})
        .drop(columns=["year"])
    )

    precinctGeoJson = gpd.read_file(join("geojson", "Police Precincts.geojson"))
    precinctGeoJson.precinct = precinctGeoJson.precinct.astype("Int32")
    precinctDF = precinctGeoJson.merge(yearCrimeCount, on="precinct", how="left")

    return createChloropleth(
        map_=map_,
        geoData=precinctDF,
        columns=["precinct", "count"],
        keyOn="feature.properties.precinct",
    )
