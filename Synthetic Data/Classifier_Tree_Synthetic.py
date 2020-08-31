import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import Counter

# Features must be DataFrame
# Ground Truth as seperated array

# Creating the model we want to use -
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}


# Creating the model - (for our model : C=50.0, kernel='linear', gamma=1e-09)
def set_SVM_model(features, ground_truth, C=50.0, kernel='linear', gamma=1e-09, random_state=10):
    class_weights = get_class_weights(ground_truth)
    model = svm.SVC(C, kernel, gamma, random_state, class_weight=class_weights)
    model = model.fit(features, ground_truth)
    predictions = model.predict(features)
    return predictions, model


# Split to two child nodes -
def get_split(features, ground_truth):
    predictions, model = set_SVM_model(features, ground_truth)
    right_group = features[predictions == 1]
    right_group_Y = ground_truth[predictions == 1]
    left_group = features[predictions == -1]
    left_group_Y = ground_truth[predictions == -1]
    groups = left_group, right_group
    new_ground_truth = left_group_Y, right_group_Y
    # Calculate the error -
    accuracy = accuracy_score(ground_truth, predictions)
    error_rate = 1 - accuracy
    return {'left_records': left_group.index, 'right_records':right_group.index, 'groups': groups,
            'ground truth': new_ground_truth, 'model': model, 'error': error_rate}


# Create child splits for a node or make terminal -
def split(node, index, alpha, records_per_leaf, min_size_leaf, model_list, left_list=[1, -1, -1], right_list=[2, -1, -1]):
    left, right = node['groups']
    left_Y , right_Y = node['ground truth']
    del (node['groups'])
    # process left child
    index_l = left_list[index]
    node_left = get_split(left, left_Y)
    error_l = node_left['error']
    exp = len(left)
    if exp <= min_size_leaf and (0.5-alpha) <= error_l <= (0.5+alpha):
        left_list[index_l] = -1
        right_list[index_l] = -1
        model_list[index_l] = 0
        records_per_leaf[index_l] = node['left_records']
    else:
        #node['left'] = get_split(left, left_Y)
        node_number = max(right_list)
        left_list[index_l] = node_number + 1
        right_list[index_l] = node_number + 2
        model_list[index_l] = node_left['model']
        left_list.extend([-1,-1])
        right_list.extend([-1,-1])
        model_list.extend([0,0])
        split(node_left, index_l, alpha, records_per_leaf, min_size_leaf, model_list, left_list, right_list)

    # process right child
    index_r = right_list[index]
    node_right = get_split(right, right_Y)
    error_r = node_right['error']
    exp2 = len(right)
    if exp2 <= min_size_leaf and (0.5-alpha) <= error_r <= (0.5+alpha):
        left_list[index_r] = -1
        right_list[index_r] = -1
        model_list[index_r] = 0
        records_per_leaf[index_r] = node['right_records']
    else:
        #node['right'] = get_split(right, right_Y)
        node_number = max(right_list)
        left_list[index_r] = node_number + 1
        right_list[index_r] = node_number + 2
        model_list[index_r] = node_right['model']
        left_list.extend([-1,-1])
        right_list.extend([-1,-1])
        model_list.extend([0,0])
        split(node_right, index_r, alpha, records_per_leaf, min_size_leaf, model_list, left_list, right_list)

    return left_list, right_list, model_list, records_per_leaf


# Build a decision tree
def build_tree(features, ground_truth, alpha, min_size_leaf, model_list=[0, 0, 0]):
    root = get_split(features, ground_truth)
    index = 0
    model_list[index] = root['model']
    records_per_leaf = {}
    lists = split(root, index, alpha, records_per_leaf, min_size_leaf, model_list)
    return lists


N_records = 10000 # number of records per population - total records is double than this
min_size_leaf = N_records
alpha = 0.1  # The deviation we allow from 0.5 error

