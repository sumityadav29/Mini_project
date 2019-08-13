from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import pandas as pd
from sklearn import metrics
from Confuse import main
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

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

knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30, weights='uniform')
knn.fit(X, Y)
#nn = NearestNeighbors(n_neighbors=10, algorithm='auto', leaf_size=30)
#nn.fit(X, Y)

y_scores = knn.predict_proba(X2)
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

Y_pred = knn.predict(X2)
err = 100 - metrics.mean_absolute_error(Y2, Y_pred) * 100

for i in (0, len(Y_pred)-1) :
	Y_pred[i] = int(Y_pred[i])

#acc = metrics.accuracy_score(Y2, knn.predict(X2));

#print (type(Y_pred[1]))
print ("Accuracy : %f" % err)
#print (acc)
#main(Y2, knn.predict(X2))
print('Precision - '+str(metrics.precision_score(Y2, Y_pred.round()) * 100))
print('Recall - '+str(metrics.recall_score(Y2, Y_pred.round()) * 100))
