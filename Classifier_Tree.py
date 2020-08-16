import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from collections import Counter

# Creating the model we want to use -

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}


# Creating the model - (for our model : C=50.0, kernel='linear', gamma=1e-09)
def set_SVM_model(features, ground_truth, C=50.0, kernel='linear', gamma=1e-09, random_state=10):
    class_weights = get_class_weights(ground_truth)
    model = svm.SVC(C, kernel, gamma, random_state, class_weight = class_weights)
    model = model.fit(features, ground_truth)
    predictions = cross_val_predict(model, features, ground_truth, cv=10)
    return predictions, model


# Split to two child nodes -
def get_split(features, ground_truth):
    predictions, model = set_SVM_model(features, ground_truth)
    right_group = features[predictions == 1]
    right_group_Y = ground_truth[predictions == 1]
    left_group = features[predictions == -1]
    left_group_Y = ground_truth[predictions == -1]
    groups = left_group, right_group
    ground_truth = left_group_Y, right_group_Y
    # Calculate the error -
    accuracy = accuracy_score(ground_truth, predictions)
    error_rate = 1 - accuracy
    return {'left_records': left_group.index, 'right_records':right_group.index, 'groups': groups,
            'ground truth': ground_truth, 'model': model, 'error': error_rate}


# Create child splits for a node or make terminal -
def split(node, alpha, left_list, right_list, model_list, index, min_size_leaf=None):
    left, right = node['groups']
    left_Y , right_Y = node['ground truth']
    del (node['groups'])
    # process left child
    index_l = left_list[index]
    if len(left) <= min_size_leaf or node['error'] <= (0.5+alpha):
        left_list[index_l] = -1
        right_list[index_l] = -1
        model_list[index_l] = 0
    else:
        node['left'] = get_split(left, left_Y)
        node_number = max(right_list)
        left_list[index_l] = node_number + 1
        right_list[index_l] = node_number + 2
        model_list[index_l] = node['model']
        left_list.extend([-1,-1])
        right_list.extend([-1,-1])
        model_list.extend([0,0])
        split(node['left'], alpha, left_list, right_list, model_list, index_l, min_size_leaf)

    # process right child
    index_r = right_list[index]
    if len(right) <= min_size_leaf or node['error'] <= (0.5 + alpha):
        left_list[index_r] = -1
        right_list[index_r] = -1
        model_list[index_r] = 0
    else:
        node['right'] = get_split(right, right_Y)
        node_number = max(right_list)
        left_list[index_r] = node_number + 1
        right_list[index_r] = node_number + 2
        model_list[index_r] = node['model']
        left_list.extend([-1,-1])
        right_list.extend([-1,-1])
        model_list.extend([0,0])
        split(node['right'], alpha, left_list, right_list, model_list, index_r, min_size_leaf)

    return left_list, right_list, model_list


# Build a decision tree
def build_tree(features, ground_truth, left_list, right_list, model_list, alpha, min_size_leaf=None):
    root = get_split(features, ground_truth)
    index = 0
    model_list[index] = root['model']
    lists = split(root, alpha, left_list, right_list, model_list, min_size_leaf,index)
    return lists


def synthetic_data(N):
    # N is the no. of samples in each population

    mu = [(-300, 0), (100, 60), (0, -200)]
    sigma = 1
    P_A = [0.35, 0.55, 0.1]
    P_B = [0.5, 0.1, 0.4]

    popA_1 = np.random.normal(mu[0], sigma, (int(N * P_A[0]), 2))
    popA_2 = np.random.normal(mu[1], sigma, (int(N * P_A[1]), 2))
    popA_3 = np.random.normal(mu[2], sigma, (int(N * P_A[2]), 2))
    popA = np.concatenate((popA_1, popA_2, popA_3))

    popB_1 = np.random.normal(mu[0], sigma, (int(N * P_B[0]), 2))
    popB_2 = np.random.normal(mu[1], sigma, (int(N * P_B[1]), 2))
    popB_3 = np.random.normal(mu[2], sigma, (int(N * P_B[2]), 2))
    popB = np.concatenate((popB_1, popB_2, popB_3))

    # Creating DataFrame of the populations and the labels to shuffle the order -
    Y = np.concatenate((np.ones(N), -1 * np.ones(N)))  # 1 for popA and -1 for popB
    X = np.concatenate((popA, popB))
    norm_X = normalize(X)

    return X, norm_X, Y


min_size_leaf = 1000
alpha = 0.06  # The deviation we allow from 0.5 error
left_list = [1, -1, -1]
right_list = [2, -1, -1]
model_list = [0, 0, 0]


X, norm_X, Y = synthetic_data(10000)
lists = build_tree(norm_X, Y, left_list, right_list, model_list, min_size_leaf, alpha)