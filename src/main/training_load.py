import pandas as pd
import sklearn.cluster._kmeans as kmeans
import pickle
import os

class TrainingLoad():
    columns = [
        'Elapsed Time',
        'Moving Time',
        'Distance',
        'Max Speed',
        'Average Speed',
        'Elevation Gain',
        'Elevation Loss',
        'Max Heart Rate',
        'Average Heart Rate',
        'Max Watts',
        'Average Watts',
        'Max Temperature',
        'Average Temperature',
        'Total Work',
        'Weighted Average Power',
        'Weather Temperature',
        'Apparent Temperature',
        'Humidity',
        'Wind Speed',
        'Wind Gust',
        'UV Index'
    ]
    default_values = {
        'Elapsed Time': 0,
        'Moving Time': 0,
        'Distance': 0,
        'Max Speed': 0,
        'Average Speed': 0,
        'Elevation Gain': 0,
        'Elevation Loss': 0,
        'Max Heart Rate': 0,
        'Average Heart Rate': 0,
        'Max Watts': 0,
        'Average Watts': 0,
        'Max Temperature': 0,
        'Average Temperature': 0,
        'Total Work': 0,
        'Weighted Average Power': 0,
        'Weather Temperature': 0,
        'Apparent Temperature': 0,
        'Humidity': 0,
        'Wind Speed': 0,
        'Wind Gust': 0,
        'UV Index': 0
    }
    def __init__(self, split: float):
        self.split = split
        self.model = None
    
    def initModel(self, clusters=8):
        self.model = kmeans.KMeans(n_clusters=clusters)

    def read_in_csv(self):
        data = pd.read_csv('activities.csv')
        data = data[data['Activity Type'] == 'Run']
        data.fillna(self.default_values, inplace=True)
        samples = data.shape[0]
        trainNum = int(samples * self.split)
        return (data[self.columns][:trainNum], data[self.columns][trainNum:])

    def fitWithData(self, xTrain):
        self.model.fit(xTrain)
        self.saveModel()
    
    def predictLabel(self, val):
        if not self.model:
            self.model = self.loadModel()
        return self.model.predict(val)

    def saveModel(self, filePath = os.path.abspath(os.path.dirname(f"{__file__}_model.pkl"))):
        with open(filePath, 'rb') as f:
            pickle.dump(self.model, f)
    
    def loadModel(self, filePath = os.path.abspath(os.path.dirname(f"{__file__}_model.pkl"))):
        if os.path.isfile(filePath):
            with open(filePath, 'rb') as f:
                self.model = pickle.load(f)
    
    def hasSavedModel(self, filePath = os.path.abspath(os.path.dirname(f"{__file__}_model.pkl"))):
        return os.path.exists(filePath)
    
    def export(self):
        info = {
            'section': self.predictLabel()
        }
        return info

def main():
    tl = TrainingLoad(split=0.8)
    trainX, testX = tl.read_in_csv()
    tl.fitWithData(trainX)
    tl.predictLabel(testX)

if __name__ == '__main__':
    main()