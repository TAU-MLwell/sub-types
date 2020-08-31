import numpy as np
import pandas as pd


# importing the data set -
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
