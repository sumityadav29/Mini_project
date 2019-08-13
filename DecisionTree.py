import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

#from Confuse import main

X = pd.read_csv('Data/Train/Train_Combine.csv', usecols=[
                'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])
Y = pd.read_csv('Data/Train/Train_Combine.csv', usecols=['PM 2.5'])

X = X.values
Y = Y.values

X2 = pd.read_csv('Data/Test/Test_Combine.csv', usecols=[
                 'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])
Y2 = pd.read_csv('Data/Test/Test_Combine.csv', usecols=['PM 2.5'])

X2 = X2.values
Y2 = Y2.values

regr_1 = DecisionTreeRegressor(max_depth=15)
regr_1.fit(X, Y)


y_1 = regr_1.predict(X2)

y_scores = regr_1.predict(X2)
fpr, tpr, threshold = roc_curve(Y2, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()



print (100 - mean_absolute_error(Y2, y_1)*100)

print ('Precision - ' + str(metrics.precision_score(Y2,y_1.round()) * 100))
print ('Recall - ' + str(metrics.recall_score(Y2,y_1.round()) * 100))

export_graphviz(regr_1, out_file = 'tree', feature_names = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])

#main(Y2, y_1)

#plt.figure()
#plt.scatter(Y, Y, c="k", label="data")
#plt.plot(Y2, y_1, c="g", label="max_depth=2", linewidth=2)
#plt.plot(X2, y_1, c="r", label="max_depth=5", linewidth=2)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()
