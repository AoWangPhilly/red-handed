from crime_data import CrimeData

class Dashboard():
    def __init__(self, files):
        self.files = files
    
    def getFiles(self):
        return self.files
    
    def setFiles(self, files):
        self.files = files
    