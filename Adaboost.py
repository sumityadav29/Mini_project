import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
dataframe = pd.read_csv("Data/Original-Data/Original_Combine.csv")
array = dataframe.values
b = np.ones((1155,1))
array = np.hstack((array,b))

for i in range(0,1155):
	if array[i,8] <= 30:
		array[i,9] = 1
	elif array[i,8] <= 60:
		array[i,9] = 2
	elif array[i,8] <= 90:
		array[i,9] = 3
	elif array[i,8] <= 120:
		array[i,9] = 4
	elif array[i,8] <= 250:
		array[i,9] = 5
	else:
		array[i,9] = 6

X = array[:,0:9]
y = array[:,9]

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, encoded, cv=kfold)
#print(results.mean())

#seed = 7
#num_trees = 100
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
#results = model_selection.cross_val_score(model, X, encoded, cv=kfold)
#print(results.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy - ',metrics.accuracy_score(y_test, y_pred)*100)
print('Precision - ' + str(metrics.precision_score(y, y_pred.round()) * 100))
