import numpy as np
import pandas as pd


class CrimeData():
    def __init__(self, file="data/incidents2020.csv"):
        self.file = file
        self.df = pd.read_csv(file)

    def __str__(self):
        idx = self.getFile().find("20")
        year = self.getFile()[idx:idx+4]
        string = "Philadelphia Crime Data: {}\n{}".format(
            year, self.getDataFrame().head())
        return string

    def clean(self):
        pass

    def eachCrime(self):
        return self.getDataFrame().text_general_code.value_counts()

    def getFile(self):
        return self.file

    def setFile(self, file):
        self.file = file

    def getDataFrame(self):
        return self.df


if __name__ == "__main__":
    print(CrimeData().eachCrime())
