import pandas as pd
import numpy as np
import datetime
import json
from sodapy import Socrata


def create_schema():
    """Creates a schema that describes all the dataset's columns

    Returns:
        pd.core.DataFrame: The dataset schema
    """

    crime_schema = pd.read_csv(
        'NYPD_Incident_Level_Data_Column_Descriptions.csv', index_col='Column')
    return crime_schema


def get_user_cred():
    """Accesses my app token and login info for the API

    Returns:
        list of str: list of token, username, and password
    """
    with open('app_token') as f:
        return f.read().split('\n')


def separate_by_yr(df):
    """Separates the full dataset into a dictionary holding year-assigned crimes

    Args:
        df (pd.core.DataFrame): Raw dataframe with mixed dates

    Returns:
        dictionary of dataframes: the crimes assigned by year
    """
    year_dict = {}

    # Grab only the year and make it an int type
    out_df = df.copy(deep=True)
    out_df['Year'] = out_df['CMPLNT_TO_DT'].dt.year

    # Get only 2006-2019
    latest_yr, earliest_yr = 2006, 2020
    out_df = out_df[(out_df['Year'] > latest_yr) &
                    (out_df['Year'] < earliest_yr)]
    for yr in range(latest_yr, earliest_yr):
        select_yr = out_df[out_df['Year'] == yr]
        year_dict[yr] = select_yr

    return year_dict


def convert_str_to_dt(df):
    out_df = df.copy(deep=True)
    out_df[['CMPLNT_TO_DT', 'CMPLNT_FR_DT', 'RPT_DT']] = \
        out_df[['CMPLNT_TO_DT', 'CMPLNT_FR_DT', 'RPT_DT']].apply(
            pd.to_datetime, format='%Y-%m-%d', errors='coerce')
    return out_df


if __name__ == '__main__':
    info = get_user_cred()
    # Get's the NYPD Complaint Data Historic
    # This dataset includes all valid felony, misdemeanor, and violation crimes reported to
    # the New York City Police Department (NYPD) from 2006 to the end of last year (2019).
    # https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
    client = Socrata('data.cityofnewyork.us',
                     info[0],
                     username=info[1],
                     password=info[2])

    # results = client.get_all("qgea-i56i")
    results = client.get('qgea-i56i')  # default 1000 rows
    crime_df_yrs = pd.DataFrame.from_records(results)
    crime_df_yrs.columns = crime_df_yrs.columns.str.upper()

    crime_df_yrs.set_index('CMPLNT_NUM', inplace=True)
    
    crime_df_yrs = convert_str_to_dt(crime_df_yrs)

    sep_year_crime = separate_by_yr(crime_df_yrs)
    for yr in range(2007, 2020):
        sep_year_crime[yr].to_csv('years/{}.csv'.format(yr))
