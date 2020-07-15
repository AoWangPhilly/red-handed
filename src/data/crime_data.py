"""
description: crime_data.py
author: ao wang
date: 07/12/2020
"""

import pandas as pd
import numpy as np


class CrimeData():
    """
    The class CrimeData is able to count the number of each crime, clean the data, 
    format as GeoJSON, and save as it.

    Attributes
    ==========
    file : str
           The crime CSV filename
    """
    def __init__(self, file="src/data/dirty/incidents2020.csv"):
        self.file = file
        self.df = pd.read_csv(file)

    def __str__(self):
        """Prints out the file name and shows the beginning of the dataframe"""
        idx = self.getFile().find("20")
        year = self.getFile()[idx:idx+4]
        string = "Philadelphia Crime Data: {}\n{}".format(
            year, self.getDataFrame().head())
        return string

    def clean(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Drops any missing crime data
        cleaned = self.df.dropna()

        # Drops duplicates
        if np.any(cleaned.duplicated):
            cleaned = cleaned.drop_duplicates()
        
        # Orders the crimes by earliest in the year
        cleaned = cleaned.sort_values(by=["dispatch_date_time"])
        return cleaned

    def eachCrime(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.getDataFrame().text_general_code.value_counts()

    def getFile(self):
        return self.file

    def setFile(self, file):
        self.file = file

    def getDataFrame(self):
        return self.df



if __name__ == "__main__":
    # for year in range(2006, 2021):
    #     fileName = "incidents{}.csv".format(year)
    #     print("Cleaning Crime Data: {}".format(year))
    #     crime = CrimeData(fileName)
    #     crime.clean().to_csv("cleaned{}".format(fileName))
    #     print("Saved Crime Data: {}".format(year))
    print(CrimeData())
