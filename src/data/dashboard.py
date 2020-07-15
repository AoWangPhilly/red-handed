"""

"""

from crime_data import CrimeData
from os import listdir
from os.path import isfile, join
import json


class Dashboard():
    """[summary]
    """

    def __init__(self, files):
        self.files = files
        dataframes = {}

        for file in self.files:
            idx = file.find("20")
            year = file[idx:idx+4]
            dataframes[year] = CrimeData(file)
            
        self.dataframes = dataframes

    def __str__(self):
        return "List of Philadelphia Crime Data in: {}".format(list(self.dataframes.keys()))

    def getFiles(self):
        return self.files

    def setFiles(self, files):
        self.files = files

    def totalCrimePerYear(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return {year: len(self.dataframes[year]) for year in self.dataframes}

    def typeOfCrimePerYear(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return {year: self.countEachCrime(year) for year in self.dataframes}

    def countEachCrime(self, year):
        """[summary]

        Args:
            year ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.dataframes.year.text_general_code.count_values()

    def weatherCorrelation(self):
        pass

    def saveAsJSON(self, data, name):
        """[summary]

        Args:
            data ([type]): [description]
            name ([type]): [description]
        """
        with open(name, "w") as j:
            j.dump(data, name)


if __name__ == "__main__":
    dir = "src/data/cleaned"
    files = ["{}/{}".format(dir, f)
             for f in listdir(dir) if isfile(join(dir, f))]
    d = Dashboard(files)
    print(d)
