import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

# PATH = 'acc_per_sampler.p'

# metrics = pickle.load(open(PATH, 'rb'))

def bootstrap_test(samples_A, samples_B, repeat=100000):
    """
    Calculate bootstrap test statistic. 
    Important: mean_A - mean_B >= 0.
    :return: p
    """
    if np.mean(samples_B) > np.mean(samples_A): return bootstrap_test(samples_B, samples_A, repeat)
    observations = np.hstack((samples_A, samples_B))
    n = len(samples_A)
    m = len(samples_B)

    t_star = np.mean(samples_A) - np.mean(samples_B)
    t = np.zeros(repeat)
    for i in range(repeat):
        # This could be a permutation instead bootstrap resampling
        #sample = np.random.choice(observations, len(observations), replace=True)
        sample = np.random.permutation(observations)
        x_star = np.mean(sample[0:n])
        y_star = np.mean(sample[n:n+m])
        t[i] = x_star - y_star

    """
    plt.hist(t)
    plt.axvline(x=t_star)
    plt.axvline(x=-t_star)
    plt.show()
    """

    p = float((t > t_star).sum() + (t < -t_star).sum()) / repeat
    return p


CLASS_MAPPING = {'cluster': 0, 'tension': 1, 'migraine': 2}
migbase = pd.read_csv('data/migbase.csv')
GROUND_TRUTH = migbase['CLASS'].map(CLASS_MAPPING)

def calculate_metrics(prediction_file):
    pred_df = pd.read_csv(prediction_file)
    pred_df = pred_df.drop('Unnamed: 0', axis=1)
    predicted_classes = np.argmax(pred_df.values, axis=1)
    accuracy = accuracy_score(GROUND_TRUTH, predicted_classes)
    kappa = cohen_kappa_score(GROUND_TRUTH, predicted_classes)

    return accuracy, kappa


print(calculate_metrics('output/oversampling/ADASYN/preds_61007.csv'))

"""
from scipy.stats import ttest_ind, ttest_rel

for sampler1 in metrics:
    for sampler2 in metrics:
        if sampler1 != sampler2:
        	#if ('WF' in sampler2 and sampler1 == 'original'):
            #print(ttest_ind(metrics[sampler1], metrics[sampler2]))
            #print(ttest_rel(metrics[sampler1], metrics[sampler2]))
	            print('{} vs. {}: p={}, mean_A,std_A={},{}; mean_B,std_B={},{}'.format(sampler1, sampler2, 
                	                                                                   bootstrap_test(metrics[sampler1], metrics[sampler2]),
                	                                                                   np.mean(metrics[sampler1]), np.std(metrics[sampler1]),
                                                                                       np.mean(metrics[sampler2]), np.std(metrics[sampler2])))
"""