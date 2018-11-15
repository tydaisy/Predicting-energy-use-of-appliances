import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# load the data
dataset = pd.read_csv('energydata_complete.csv')
features = pd.read_csv("energydata_complete.csv",nrows = 0) # fields or features of the dataset
print (dataset.corr('pearson'))

# Extract inputs and outputs
data = dataset.values
y = data[:,0:1] # outputs:the column of Appliances
x = data[:,1:30] # inputs: other features

# Data normalisation(0-1)
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)

# Feature selection
test = SelectKBest(score_func=chi2, k=21)
fit = test.fit(x, y)
np.set_printoptions(precision=3)
x = fit.transform(x)

#show the distribution of each feature one page by one page
for f in features:
	column = dataset[f]
	column.hist(color='k')
	pyplot.title(f)
	pyplot.show('r')

# show the distribution of each field in one page
dataset.hist(color='k')
pyplot.show('r')
