import pandas as pd
import numpy as np
import json
from sodapy import Socrata


def create_schema():
    crime_schema = pd.read_csv(
        'NYPD_Incident_Level_Data_Column_Descriptions.csv', index_col='Column')
    return crime_schema


def get_user_cred():
    with open('app_token') as f:
        return f.read().split('\n')


def seperate_by_yr(df):
    year_dict = {}

    # Grab only the year and make it an int type
    out_df = df.copy(deep=True)
    out_df['Year'] = out_df['CMPLNT_FR_DT'].str[6:].fillna(0)
    out_df['Year'] = out_df['Year'].astype('int16')

    # Get only 2006-2019
    latest_yr, earliest_yr = 2006, 2020
    out_df = out_df[out_df['Year'] > latest_yr]
    for yr in range(latest_yr, earliest_yr):
        select_yr = out_df[out_df['Year'] == yr]
        year_dict[yr] = select_yr

    return year_dict


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
    crime_df_yrs.set_index('CMPLNT_NUM', inplace=True)

    sep_year_crime = seperate_by_yr(crime_df_yrs)
    for yr in range(2007, 2020):
        sep_year_crime[yr].to_csv('years/{}.csv'.format(yr))