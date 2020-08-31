import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from collections import Counter

### Features must be in a DataFrame ###

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
    predictions = cross_val_predict(model, features, ground_truth, cv=10)
    return predictions, model


# Split to two child nodes -
def get_split(features, ground_truth):
    predictions, model = set_SVM_model(features, ground_truth)
    right_group = features[predictions == 1] #Above 60y
    right_group_Y = ground_truth[predictions == 1]
    left_group = features[predictions == -1] #Under 60y
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
    error_l = round(node_left['error'],2)
    exp = len(left)
    if exp <= min_size_leaf or (0.5-alpha) <= error_l <= (0.5+alpha):
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
    error_r = round(node_right['error'],2)
    exp2 = len(right)
    if exp2 <= min_size_leaf or (0.5-alpha) <= error_r <= (0.5+alpha):
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



##### Pre-processing of the data #####

dataset = pd.read_excel('/Users/morzukin/PycharmProjects/final_project/from_web/corona_dataset.xlsx')

# Remove samples with corona_result as 'אחר' -
dataset = dataset[dataset.corona_result != 'אחר']  # remove 1391 observations
# Changing some value's name -
to_replace_1 = 'חיובי'
dataset = dataset.replace(to_replace_1,1)
to_replace_0 = 'שלילי'
dataset = dataset.replace(to_replace_0,0)
dataset = dataset.replace([1,0],['Yes','No'])

# Remove samples with no age indications -
data = dataset[~dataset['age_60_and_above'].isnull()]
data['age_60_and_above'] = data['age_60_and_above'].replace(['Yes','No'],[1,-1]) # 1 = above 60y  & -1 = under 60y

# Only positive records to Corona - population by age
positive_data = data[data['corona_result'] == 'Yes'] # 9937 records with a positive diagnosis
ground_truth = positive_data['age_60_and_above'] #1936 records above 60 years old

# One Hot Encoding for the features  -
symptoms = ['fever','sore_throat','shortness_of_breath','head_ache','test_indication'] # Witn no gender & cough
features = positive_data[symptoms]
features_dummies = pd.get_dummies(features[symptoms],dummy_na=False, drop_first=False)

##### SETTINGS #####

min_size_leaf = 2000
alpha = 0.03  # The deviation we allow from 0.5 error

outcomes = build_tree(features_dummies, ground_truth, alpha, min_size_leaf)


### Plotting the clusters ### 

from MulticoreTSNE import MulticoreTSNE as TSNE

cluster1 = outcomes[3][1]
cluster2 = outcomes[3][5]
cluster3 = outcomes[3][6]
cluster4 = outcomes[3][4]

tsne = TSNE(n_components=2, n_jobs=-1, random_state=42)
tsne_embedding = tsne.fit_transform(features_dummies)

TSNE_pd = pd.DataFrame(tsne_embedding)
TSNE_pd['index'] = ground_truth.index


# colors after division to population by age -
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

clus1 = mpatches.Patch(color='#FFCE86', label='cluster 1')
clus2 = mpatches.Patch(color='#AAA7BC', label='cluster 2')
clus3 = mpatches.Patch(color='#9DC1FF', label='cluster 3')
clus4 = mpatches.Patch(color='#EB543D', label='cluster 4')

plt.figure(figsize=(17,17))
plt.plot(TSNE_pd.loc[TSNE_pd['index'].isin(cluster1)][0], TSNE_pd.loc[TSNE_pd['index'].isin(cluster1)][1], 'o', color='#FFCE86',alpha=0.3)
plt.plot(TSNE_pd.loc[TSNE_pd['index'].isin(cluster2)][0], TSNE_pd.loc[TSNE_pd['index'].isin(cluster2)][1], 'o', color='#AAA7BC',alpha=0.3)
plt.plot(TSNE_pd.loc[TSNE_pd['index'].isin(cluster3)][0], TSNE_pd.loc[TSNE_pd['index'].isin(cluster3)][1], 'o', color='#9DC1FF',alpha=0.3)
plt.plot(TSNE_pd.loc[TSNE_pd['index'].isin(cluster4)][0], TSNE_pd.loc[TSNE_pd['index'].isin(cluster4)][1], 'o', color='#EB543D',alpha=0.3)
plt.xticks([])
plt.yticks([])
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Results of classifier')
plt.legend(handles=[clus1, clus2,clus3,clus4])
plt.show()
plt.savefig('Results of classifier')