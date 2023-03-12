![NYC](https://www.travelandleisure.com/thmb/91pb8LbDAUwUN_11wATYjx5oF8Q=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/new-york-city-evening-NYCTG0221-52492d6ccab44f328a1c89f41ac02aea.jpg)

# RedHanded - DSCI 521 Final Project

RedHanded is a project I've previous tried to work on, but was never able to complete. The old project worked on Philadelphia Crime data, but this project deals with NYC Crime data from 2006-2021.
  
The goal of the project is to create a predictor model that accurately forecasts crime rates in NYC. Additionally, I also wanted to find out if temperature had an effect on the number of crimes in a year. Finally, I wanted to create map and other visualizations to see where is crime most concentrated and when it occurs most often.

## Table of Contents

- [RedHanded - DSCI 521 Final Project](#redhanded---dsci-521-final-project)
  - [Table of Contents](#table-of-contents)
    - [Authors](#authors)
    - [Setup and Installation](#setup-and-installation)
    - [How to Run](#how-to-run)
    - [Technologies](#technologies)
    - [Data Sources](#data-sources)
    - [Code](#code)
    - [Goals](#goals)
    - [Challenges](#challenges)
    - [Limitations](#limitations)

### Authors

- [@aowang](https://github.com/AoWangPhilly)

### Setup and Installation

- This is primarly to run on MacOS M1

1. `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. `brew install openjdk@11`
3. `brew install apache-spark`
4. `python3 -m venv venv`
5. `source venv/bin/activate`
6. `pip3 install -r requirements.txt`

*** Note for Windows users, I'm not quite sure how to download Apache Spark. But once you get Java and Apache Spark, the rest of the steps should be valid

### How to Run

- After all required software and packages are installed, run `streamlit run Home.py` to run the Streamlit web application.
- To run the notebooks, there're no special setups to execute the coding cells.

### Technologies

- PySpark for big data
- Streamlit for web app development and hosting
- Pandas for data analysis and manipulation
- Folium + GeoJSON for maps
- Plotly for visualizations

### Data Sources

All links provide the data and data dictionaries for each of the fields. I also provided the data dictionaries in the `data_dictionary` folder. The GEOJSON data is in the `geojson` folder. And the rest of the data will be in `data`. The crime data is saved as a parquet file, `data/NYPD_Complaint_Data_Historic.parquet`. And the weather data is called `data/USW00094728.csv`.

- NYPD Complaint Data Historic (~3 GB CSV file)
  - <https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i>
  - Data is updated annually and provides attachments, such as:
    - Data dictionary
    - The NYPD penal codes
- Boundaries of Police Precincts (GeoJSON)
  - <https://data.cityofnewyork.us/Public-Safety/Police-Precincts/78dh-3ptz>
- Boundaries of Boroughs (GeoJSON)
  - <https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm>
- NYC Weather Data (CSV file)
  - <https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00094728/detail>
  
### Code

- All the notebooks are in the folder `notebooks`. There should be three: `Data Cleaning for Crime Data.ipynb`, `EDA for Crime Data.ipynb`, and `EDA for Weather Data.ipynb`
  
- Aside from the notebooks there are also Python scripts in the `src` folder that the Streamlit app uses to create models and visualizations.

- There are also Python scripts in the `pages` folder and a `Home.py` that acts as point for the Streamlit application.

### Goals

A lot of the questions I had in mind came to me thinking how safe it would be to live in NYC. Thinking of moving there in a few years, I'm wondering how bad it would be compared to Philadelphia. Perhaps that'll be another project in the future. 😉

1. How has NYC crime rates been over the years? Has it been increasing, decreasing, or stagnant? Are we able to predict crime rates? Perhaps linear regression or SARIMA model?
2. Is there any external factors on crime rate? After analyzing and plotting the data, there seems to be some seasonality. Perhaps there's a correlation between temperature/weather and crime rates. There's less crime during the winter months and more crime during the summer months. Perhaps precipitation and snow fall would also have some effect.
3. Given an area, such as a Borough or Police Precinct, what's the most common types of crime? And what're the frequencies for crimes to occur (monthly, daily, weekly, and hourly). Additionally, how fast are crimes being reported and how long do crimes take to occur?

### Challenges

- The initial challenge was getting the NYPD Complaint Data Historic. I naively thought I could use the API to get all the data. But later I found out that the dataset had almost 8 million rows and the entire CSV file was around 3 GB.

- I tried working with just Pandas, however, everytime I tried loading the dataset, it took a couple of minutes. And when I tried to work on the DataFrame, it took a couple seconds to do relatively simple operations. So I turned to saving the `CSV` file into a `parquet` file for faster loading and I broke the file into years. However I realized that I often had to work with all the crime data, so there was no point.

- I evenually turned to PySpark to deal with basic manipulation until the data became a size that I could convert as a DataFrame and create vizualizations.

- Currently, there are no issues. If the dataset size increases, PySpark is able to handle it, even if it went into the PB scale. Additionally, Streamlit is able to cache the dataframes once they load once, so users are able to have an enjoyable experience.

### Limitations

- The NYC Open Data Socrata API only provides 1000 rows at a time, so it will be insanely slow to download all ~8 million rows.
- Some limitations with the crime data is that victim and suspect fields are often missing. I would have liked to analyze that data and create a classifier model.
- Additionally, the crime data's violation and police code descriptions are often abbreviated, so it's not clear to what they mean exactly.
