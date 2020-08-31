import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

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
    norm_X = pd.DataFrame(norm_X)

    return X, norm_X, Y


N_records = 10000 # number of records per population - total records is double than this

X, norm_X, Y = synthetic_data(N_records)
