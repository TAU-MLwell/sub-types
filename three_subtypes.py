import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV

mu = [(-300,0),(100,60),(0,-200)]
sigma = 1
P_A = [0.35,0.55,0.1]
P_B = [0.5,0.1,0.4]
#P_A = [0,1,0]
#P_B = [0.5,0,0.5]
N = 10000 # No. of samples in each population

popA_1 = np.random.normal(mu[0], sigma, (int(N*P_A[0]),2))
popA_2 = np.random.normal(mu[1], sigma, (int(N*P_A[1]),2))
popA_3 = np.random.normal(mu[2], sigma, (int(N*P_A[2]),2))
popA = np.concatenate((popA_1,popA_2,popA_3))

popB_1 = np.random.normal(mu[0], sigma, (int(N*P_B[0]),2))
popB_2 = np.random.normal(mu[1], sigma, (int(N*P_B[1]),2))
popB_3 = np.random.normal(mu[2], sigma, (int(N*P_B[2]),2))
popB = np.concatenate((popB_1,popB_2,popB_3))

# Creating DataFrame of the populations and the labels to shuffle the order -
labels = np.concatenate((np.ones(N),-1*np.ones(N))) # 1 for popA and -1 for popB
data_only = np.concatenate((popA,popB))
Data = pd.DataFrame({'Feature1': data_only[:,0], 'Feature2': data_only[:,1], 'label': labels[:,]})
#Data = shuffle(Data)
#Data = Data.apply(np.random.permutation, axis=0)

X = pd.DataFrame({'Feature1': Data['Feature1'], 'Feature2': Data['Feature2']})
Y = Data['label']

# Test set -
test = np.random.normal((0,0),200,(1000,2))

# Classification with decision tree -
#clf = RandomForestClassifier(max_depth=3, min_samples_split=10, random_state=0)
#clf.fit(X, Y)

# SVM -
#norm_X = normalize(X)
clf = svm.SVC(C=50.0, kernel='linear',gamma='auto')
clf.fit(X,Y)

# Logistic regression -
#clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, Y)


#predictions -
#norm_test = normalize(test)
predictions = clf.predict(test)
#unique, counts = np.unique(predictions, return_counts=True)
#print(dict(zip(unique, counts)))

plt.plot(test[predictions<0,0], test[predictions<0,1], 'o', color='blue')
plt.plot(test[predictions>0,0], test[predictions>0,1], 'o', color='green')
plt.plot((mu[0][0],mu[1][0],mu[2][0]),(mu[0][1],mu[1][1],mu[2][1]),'o',color='black')
plt.show()
