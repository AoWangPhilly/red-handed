"""
description: crime_data.py
author: ao wang
date: 07/12/2020
"""

import pandas as pd
import numpy as np
from geojson import GeometryCollection, Point
import json


class CrimeData():
    """
    The class CrimeData is able to count the number of each crime, clean the data, 
    format as GeoJSON, and save as it.

    Attributes
    ==========
    file : str
           The crime CSV filename
    """
    def __init__(self, file="incidents2020.csv"):
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
        # Drops any missing crime data
        cleaned = self.df.dropna()

        # Drops duplicates
        if np.any(cleaned.duplicated):
            cleaned = cleaned.drop_duplicates()
        
        # Orders the crimes by earliest in the year
        cleaned = cleaned.sort_values(by=["dispatch_date_time"])
        return cleaned

    def eachCrime(self):
        return self.getDataFrame().text_general_code.value_counts()

    def getFile(self):
        return self.file

    def setFile(self, file):
        self.file = file

    def getDataFrame(self):
        return self.df

    def convertGeoJson(self):
        cleanDf = self.clean()
        geoList = []
        locations = cleanDf[['lat', 'lng']]
        locationList = locations.values.tolist()

        for point in range(len(locationList)):
            crimePoint = Point(locationList[point], properties={
                               "location_block": cleanDf.location_block.iloc[point],
                               "text_general_code": cleanDf.text_general_code.iloc[point],
                               "dispatch_date_time": cleanDf.dispatch_date_time.iloc[point]})
            geoList.append(crimePoint)
        return GeometryCollection(geoList)

    def saveAsGeoJSON(self):
        idx = self.getFile().find("20")
        year = self.getFile()[idx:idx+4]
        fileName = "{}.json".format(year)

        with open(fileName, "w") as crimeFile:
            json.dump(self.convertGeoJson(), crimeFile)


if __name__ == "__main__":
    def createFileName(year): return "incidents{}.csv".format(year)
    [CrimeData(createFileName(year)).saveAsGeoJSON() for year in range(2015,2021)]
