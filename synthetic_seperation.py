#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[ ]:


def synthetic_data(N):
    # N is the no. of samples in each population

    mu = [(-300,0),(100,60),(0,-200)]
    sigma = 1
    P_A = [0.35,0.55,0.1]
    P_B = [0.5,0.1,0.4]

    popA_1 = np.random.normal(mu[0], sigma, (int(N*P_A[0]),2))
    popA_2 = np.random.normal(mu[1], sigma, (int(N*P_A[1]),2))
    popA_3 = np.random.normal(mu[2], sigma, (int(N*P_A[2]),2))
    popA = np.concatenate((popA_1,popA_2,popA_3))

    popB_1 = np.random.normal(mu[0], sigma, (int(N*P_B[0]),2))
    popB_2 = np.random.normal(mu[1], sigma, (int(N*P_B[1]),2))
    popB_3 = np.random.normal(mu[2], sigma, (int(N*P_B[2]),2))
    popB = np.concatenate((popB_1,popB_2,popB_3))

    # Creating DataFrame of the populations and the labels to shuffle the order -
    Y = np.concatenate((np.ones(N),-1*np.ones(N))) # 1 for popA and -1 for popB
    X = np.concatenate((popA,popB))
    norm_X = normalize(X)
    
    return X,norm_X,Y


# In[ ]:


X,norm_X,Y = synthetic_data(10000)


# In[ ]:


# SVM -
clf = svm.SVC(C=0.01,kernel='rbf',gamma=1e-09)
clf.fit(norm_X,Y)

predictions = clf.predict(norm_X)


# In[ ]:


plt.plot(X[predictions<0,0], X[predictions<0,1], 'o', color='blue')
plt.plot(X[predictions>0,0], X[predictions>0,1], 'o', color='green')
mu = [(-300,0),(100,60),(0,-200)]
plt.plot((mu[0][0],mu[1][0],mu[2][0]),(mu[0][1],mu[1][1],mu[2][1]),'o',color='black')
plt.show()


# In[ ]:


pred_df = pd.DataFrame({'Ground Truth': Y, 'Predictions': predictions})

CM = pd.DataFrame(pred_df.groupby(['Predictions','Ground Truth']).size(),columns = [' '])
CM = CM.unstack(level=0)
CM


# In[ ]:


accuracy = accuracy_score(Y, predictions)
error_rate = 1 - accuracy
error_rate


# In[ ]:


#popA (classified as +1 - green)
X_popA = norm_X[predictions>0,:]
Y_popA = Y[predictions>0]
dfA = pd.DataFrame({'Feature1': X_popA[:,0], 'Feature2': X_popA[:,1], 'Label':Y_popA})

#Balance of the data - 
to_remove = np.random.choice(dfA[dfA['Label']==1].index,size=4500,replace=False)
A_df = dfA.drop(to_remove)

#Checking the results - 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Population A')
ax1.plot(A_df['Feature1'], A_df['Feature2'],'o', color='green')
ax1.plot(0,0,'o',color='black')
ax1.set(xlabel='Feature 1', ylabel='Feature 2')
A_labels = A_df['Label'].value_counts().sort_index()
ax2.bar(A_labels.index, A_labels.values,color='steelblue',edgecolor='black')


# In[ ]:


# Classified subpopulation A - 

clf_A = svm.SVC(C=0.01,kernel='rbf',gamma=1e-09)
clf_A.fit(A_df.iloc[:, 0:2] ,A_df['Label'])
predictions_A = clf_A.predict(A_df.iloc[:, 0:2])

predA_df = pd.DataFrame({'Ground Truth': A_df['Label'], 'Predictions': predictions_A})
CM_A = pd.DataFrame(predA_df.groupby(['Predictions','Ground Truth']).size(),columns = [' '])
CM_A = CM_A.unstack(level=0)
CM_A


# In[ ]:


accuracy = accuracy_score(A_df['Label'], predictions_A)
error_rate = 1 - accuracy
error_rate


# In[ ]:


#popB (classified as -1 - blue)
X_popB = norm_X[predictions<0,:]
Y_popB = Y[predictions<0]
dfB = pd.DataFrame({'Feature1': X_popB[:,0], 'Feature2': X_popB[:,1], 'Label':Y_popB})

#Balance of the data - 
to_remove = np.random.choice(dfB[dfB['Label']==-1].index,size=4500,replace=False)
B_df = dfB.drop(to_remove)

#Checking the results - 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Population B')
ax1.plot(B_df['Feature1'], B_df['Feature2'],'o', color='blue')
ax1.plot(0,0,'o',color='black')
ax1.set(xlabel='Feature 1', ylabel='Feature 2')
B_labels = B_df['Label'].value_counts().sort_index()
ax2.bar(B_labels.index, B_labels.values,color='steelblue',edgecolor='black')


# In[ ]:


# Classified subpopulation B - 

clf_B = svm.SVC(C=0.01,kernel='rbf',gamma=1e-09)
clf_B.fit(B_df.iloc[:, 0:2] ,B_df['Label'])
predictions_B = clf_B.predict(B_df.iloc[:, 0:2])

predB_df = pd.DataFrame({'Ground Truth': B_df['Label'], 'Predictions': predictions_B})
CM_B = pd.DataFrame(predB_df.groupby(['Predictions','Ground Truth']).size(),columns = [' '])
CM_B = CM_B.unstack(level=0)
CM_B


# In[ ]:


accuracy = accuracy_score(B_df['Label'], predictions_B)
error_rate = 1 - accuracy
error_rate


# In[ ]:


feature1 = B_df['Feature1']
feature2 = B_df['Feature2']

plt.plot(feature1[predictions_B==-1], feature2[predictions_B==-1], 'o', color='yellow')
plt.plot(feature1[predictions_B==1], feature2[predictions_B==1], 'o', color='orange')
plt.plot(0,0,'o',color='black')
plt.show()

