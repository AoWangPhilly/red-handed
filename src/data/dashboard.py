"""

"""

from crime_data import CrimeData
from os import listdir
from os.path import isfile, join
import json
from pprint import pprint
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from numpyencoder import NumpyEncoder
from uszipcode import SearchEngine, Zipcode


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

    def totalCrimesPerMonth(self, year):
        numToMonth = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                      7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        df = self.dataframes[year]
        df.dispatch_date_time = pd.to_datetime(df.dispatch_date_time)
        m = df.dispatch_date_time.dt.month
        return {numToMonth.get(month): int(m[m == month].count()) for month in range(1, 13)}

    def totalCrimesPerMonthEachYear(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return {year: self.totalCrimesPerMonth(year) for year in self.dataframes}

    def saveAsJSON(self, data, name):
        """[summary]

        Args:
            data ([type]): [description]
            name ([type]): [description]
        """
        with open(name, "w") as j:
            json.dump(data, j, cls=NumpyEncoder)

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

        model = LinearRegression().fit(x, y)

        r_sq = model.score(x, y)
        print('coefficient of determination:', r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        return np.ceil(model.predict([[year]]))

    def getZipcode(self, lat, lng):
        search = SearchEngine()
        return search.by_coordinates(lat, lng)[0].zipcode

    def addZipcodeCol(self, year):
        crimes = self.dataframes[year]
        crimes["zipcode"] = crimes.apply(
            lambda crime: self.getZipcode(crime.lat, crime.lng), axis=1)
        crimes.to_csv(
            "/Users/aowang/red-handed/src/data/cleaned/cleanedincidents{}.csv".format(year))

    def crimesPerZipcode(self, year):
        return self.dataframes[year].zipcode.value_counts()

    # population_density

    def crimePopDensity(self, year):
        zipcodes = self.crimesPerZipcode(year).keys()
        search = SearchEngine()
        return {zipcode: search.by_prefix(zipcode).population_density for zipcode in zipcodes}

    # median_household_income
    def crimeHouseHoldIncome(self, year):
        zipcodes = self.crimesPerZipcode(year).keys()
        search = SearchEngine()
        return {zipcode: search.by_prefix(zipcode).median_household_income for zipcode in zipcodes}


if __name__ == "__main__":
    dir = "/Users/aowang/red-handed/src/data/cleaned"
    files = sorted(["{}/{}".format(dir, f)
                    for f in listdir(dir) if isfile(join(dir, f))])
    files = files[:len(files)-1]

    d = Dashboard(files)
    d.addZipcodeCol(2019)

    # crimeDict = dict(d.countEachCrime(2019))
    # d.saveAsJSON(
    #     crimeDict, "/Users/aowang/red-handed/src/data/dashboard/typeOfCrimes2019.json")

    # Total crimes each month in 2019
    # month = d.totalCrimesPerMonth(2019)
    # d.saveAsJSON(month, "src/data/dashboard/monthlyCrime.json")

    # Total crimes 2006-2020
    # crime = d.totalCrimePerYear()
    # crime[2020] = d.predictCrime(2020)[0]
    # d.saveAsJSON(crime, "src/data/dashboard/crimePerYear.json")
