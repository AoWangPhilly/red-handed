"""

"""

from crime_data import CrimeData
from os import listdir
from os.path import isfile, join
import json
from pprint import pprint
import numpy as np
from sklearn.linear_model import LinearRegression


class Dashboard():
    """[summary]
    """

    def __init__(self, files):
        self.files = files
        dataframes = {}

        for file in sorted(self.files):
            idx = file.find("20")
            year = int(file[idx:idx+4])
            dataframes[year] = CrimeData(file).clean()

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
        return self.dataframes[year].text_general_code.value_counts()

    def weatherCorrelation(self):
        pass

    def totalCrimesPerMonth(self, year):
        return {}
    
    def totalCrimesPerMonthEachYear(self):
        pass

    def saveAsJSON(self, data, name):
        """[summary]

        Args:
            data ([type]): [description]
            name ([type]): [description]
        """
        with open(name, "w") as j:
            json.dump(data, j)

    def predictCrime(self, year):
        """[summary]

        Args:
            year ([type]): [description]

        Returns:
            [type]: [description]
        """
        crime = self.totalCrimePerYear()
        numCrime = list(crime.values())
        years = list(crime.keys())
        x = np.array(years).reshape((-1, 1))
        y = np.array(numCrime)

        model = LinearRegression()
        model.fit(x, y)
        model = LinearRegression().fit(x, y)

        r_sq = model.score(x, y)
        print('coefficient of determination:', r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        return np.ceil(model.predict([[year]]))


if __name__ == "__main__":
    dir = "src/data/cleaned"
    files = sorted(["{}/{}".format(dir, f)
             for f in listdir(dir) if isfile(join(dir, f))])
    files = files[:len(files)-1]
    d = Dashboard(files)
    print(d)
    pprint(d.totalCrimePerYear())
    crime = d.totalCrimePerYear()
    crime[2020] = d.predictCrime(2020)[0]
    pprint(crime)
    d.saveAsJSON(crime, "src/data/dashboard/crimePerYear.json")
    print(d.predictCrime(2020))
    # pprint(d.typeOfCrimePerYear())
