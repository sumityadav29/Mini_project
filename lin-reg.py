import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from Confuse import main


X = pd.read_csv('Data/Train/Train_Combine.csv', usecols=[
                'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])
y = pd.read_csv('Data/Train/Train_Combine.csv', usecols=['PM 2.5'])

X = np.array(X)
y = np.array(y)

lin = linear_model.LinearRegression()

lin.fit(X, y)

X2 = pd.read_csv('Data/Test/Test_Combine.csv', usecols=[
                 'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])
Y2 = pd.read_csv('Data/Test/Test_Combine.csv', usecols=['PM 2.5'])

X2 = X2.values
Y2 = Y2.values

preds = lin.predict(X2)

err = 100 - (metrics.mean_absolute_error(Y2, preds) * 100)
print ("Accuracy: %f" % err)
main(Y2, preds)

print('Prcision - ' + str(metrics.precision_score(Y2, preds.round()) * 100))
print('Recall - ' + str(metrics.recall_score(Y2, preds.round()) * 100))
