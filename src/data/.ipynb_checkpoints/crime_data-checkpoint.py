import pandas as pd
import numpy as np
import geojson


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
        pass

    def saveAsGeoJSON(self):
        pass


if __name__ == "__main__":
    print(CrimeData().eachCrime())
