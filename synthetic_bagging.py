#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score
from imblearn.ensemble import BalancedBaggingClassifier


# In[2]:


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


# In[3]:


X,norm_X,Y = synthetic_data(10000)


# In[4]:


# Bagging model with Undersamling of the majority class -
#clf = svm.SVC(kernel='rbf',gamma=1e-09,class_weight=class_weights)
svm = svm.SVC(kernel='rbf',gamma='auto')
clf = BalancedBaggingClassifier(base_estimator=svm , n_estimators=50, max_samples=2000)
clf.fit(norm_X,Y)

predictions = clf.predict(norm_X)


# In[5]:


plt.plot(X[predictions<0,0], X[predictions<0,1], 'o', color='blue')
plt.plot(X[predictions>0,0], X[predictions>0,1], 'o', color='green')
mu = [(-300,0),(100,60),(0,-200)]
plt.plot((mu[0][0],mu[1][0],mu[2][0]),(mu[0][1],mu[1][1],mu[2][1]),'o',color='black')
plt.show()


# In[6]:


pred_df = pd.DataFrame({'Ground Truth': Y, 'Predictions': predictions})

CM = pd.DataFrame(pred_df.groupby(['Predictions','Ground Truth']).size(),columns = [' '])
CM = CM.unstack(level=0)
CM


# In[7]:


accuracy = accuracy_score(Y, predictions)
error_rate = 1 - accuracy
error_rate


# In[8]:


#popA (classified as +1 - green)
X_popA = norm_X[predictions>0,:]
Y_popA = Y[predictions>0]

#Checking the results - 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Population A')
ax1.plot(X_popA[:,0], X_popA[:,1],'o', color='green')
ax1.plot(0,0,'o',color='black')
ax1.set(xlabel='Feature 1', ylabel='Feature 2')

unique, counts = np.unique(Y_popA, return_counts=True)
A_dict = dict(zip(unique, counts))
ax2.bar(A_dict.keys(), A_dict.values(),color='steelblue',edgecolor='black')


# In[9]:


# Classified subpopulation A - 
clf_A = BalancedBaggingClassifier(base_estimator=svm , n_estimators=50, max_samples=2000)
clf_A.fit(X_popA ,Y_popA)
predictions_A = clf_A.predict(X_popA)

predA_df = pd.DataFrame({'Ground Truth': Y_popA, 'Predictions': predictions_A})
CM_A = pd.DataFrame(predA_df.groupby(['Predictions','Ground Truth']).size(),columns = [' '])
CM_A = CM_A.unstack(level=0)
CM_A


# In[10]:


accuracy = accuracy_score(Y_popA, predictions_A)
error_rate = 1 - accuracy
error_rate


# In[11]:


#popB (classified as -1 - blue)
X_popB = norm_X[predictions<0,:]
Y_popB = Y[predictions<0]

#Checking the results - 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Population B')
ax1.plot(X_popB[:,0], X_popB[:,1],'o', color='blue')
ax1.plot(0,0,'o',color='black')
ax1.set(xlabel='Feature 1', ylabel='Feature 2')

unique, counts = np.unique(Y_popB, return_counts=True)
B_dict = dict(zip(unique, counts))
ax2.bar(B_dict.keys(), B_dict.values(),color='steelblue',edgecolor='black')


# In[12]:


# Classified subpopulation B - 
clf_B = BalancedBaggingClassifier(base_estimator=svm , n_estimators=50, max_samples=2000)
clf_B.fit(X_popB ,Y_popB)
predictions_B = clf_B.predict(X_popB)

predB_df = pd.DataFrame({'Ground Truth': Y_popB, 'Predictions': predictions_B})
CM_B = pd.DataFrame(predB_df.groupby(['Predictions','Ground Truth']).size(),columns = [' '])
CM_B = CM_B.unstack(level=0)
CM_B


# In[13]:


accuracy = accuracy_score(Y_popB, predictions_B)
error_rate = 1 - accuracy
error_rate


# In[14]:


feature1 = X_popB[:,0]
feature2 = X_popB[:,1]

plt.plot(feature1[predictions_B==-1], feature2[predictions_B==-1], 'o', color='yellow')
plt.plot(feature1[predictions_B==1], feature2[predictions_B==1], 'o', color='orange')
plt.plot(0,0,'o',color='black')
plt.show()

