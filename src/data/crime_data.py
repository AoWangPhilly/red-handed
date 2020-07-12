import pandas as pd
import numpy as np
from geojson import GeometryCollection, Point


class CrimeData():
    def __init__(self, file="incidents2020.csv"):
        self.file = file
        self.df = pd.read_csv(file)

    def __str__(self):
        idx = self.getFile().find("20")
        year = self.getFile()[idx:idx+4]
        string = "Philadelphia Crime Data: {}\n{}".format(
            year, self.getDataFrame().head())
        return string

    def clean(self):
        cleaned = self.df.dropna()
        if np.any(cleaned.duplicated):
            cleaned = cleaned.drop_duplicates()
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
        locations = df[['lat', 'lng']]
        locationList = locations.values.tolist()
        for point in range(len(locationList)):
            crimePoint = Point(locationList[point], properties={
                               "location_block": cleanDf.location_block.iloc[point],
                               "text_general_code": cleanDf.text_general_code.iloc[point],
                               "dispatch_date_time": cleanDf.dispatch_date_time.iloc[point]})
            geoList.append(crimePoint)
        return GeometryCollection(geoList)

    def saveAsGeoJSON(self):
        pass


if __name__ == "__main__":
    print(CrimeData().convertGeoJson())
