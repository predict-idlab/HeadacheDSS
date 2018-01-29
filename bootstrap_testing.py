import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from terminaltables import AsciiTable
from sklearn.metrics import cohen_kappa_score, accuracy_score

def bootstrap_test(samples_A, samples_B, repeat=10, plot=False):
    """
    Calculate bootstrap test statistic. 
    Important: mean_A - mean_B >= 0.
    :return: p
    """

    # Make sure mean_A > mean_B 
    if np.mean(samples_B) > np.mean(samples_A): 
        return bootstrap_test(samples_B, samples_A, repeat)

    # Stack together the observations (which will be shuffled)
    observations = np.hstack((samples_A, samples_B))
    n = len(samples_A)
    m = len(samples_B)

    # Calculate difference of given population means
    # NULL HYPOTHESIS: 'A has larger mean due to sampling'
    t_star = np.mean(samples_A) - np.mean(samples_B)
    t = np.zeros(repeat)
    for i in range(repeat):
        # This could be a permutation instead bootstrap resampling
        # sample = np.random.permutation(observations)
        sample = np.random.choice(
            observations, 
            len(observations), 
            replace=True
        )
        x_star = np.mean(sample[0:n])
        y_star = np.mean(sample[n:n+m])
        t[i] = x_star - y_star

    if plot:
        plt.hist(t)
        plt.axvline(x=t_star)
        plt.axvline(x=-t_star)
        plt.show()

    # Calculate p-value (#resamplings that produced larger difference)
    p = float((t > t_star).sum() + (t < -t_star).sum()) / repeat
    return p


MODE = 'oversampling' # Choose 'oversampling', 'features' or 'both'
METRIC = 'accuracy'

# Create the ground truth
migbase = pd.read_csv('data/migbase.csv')
CLASS_MAPPING = {'cluster': 0, 'tension': 1, 'migraine': 2}
GROUND_TRUTH = migbase['CLASS'].map(CLASS_MAPPING)

def calculate_metrics(prediction_file):
    # Read in prediction file, which contains probabilities for all
    # classes. Take argmax as prediction. Calculate accuracy or kappa
    pred_df = pd.read_csv(prediction_file)
    pred_df = pred_df.drop('Unnamed: 0', axis=1)
    predicted_classes = np.argmax(pred_df.values, axis=1)
    if METRIC == 'accuracy':
        return accuracy_score(GROUND_TRUTH, predicted_classes)
    if METRIC == 'kappa':
        return cohen_kappa_score(GROUND_TRUTH, predicted_classes)

# Depending on the mode, iterate over directories in output/
# Calculate accuracy or kappa metrics, add them to a dict
# and apply bootstrap testing
if MODE == 'both':
    pass
else:
    root = 'output/'+MODE+'/'
    metrics = {}
    for algorithm in os.listdir(root):
        metrics[algorithm] = []
        for _file in os.listdir(root+algorithm):
            metrics[algorithm].append(calculate_metrics(root+algorithm+os.sep+_file))

    table_data = [['', METRIC]]
    for sampler1 in metrics:
        table_row = [sampler1, '{}+/-{}'.format(np.round(np.mean(metrics[sampler1]), 6),
                                                np.round(np.std(metrics[sampler1]), 4))]
        table_data.append(table_row)
        for sampler2 in metrics:
            if sampler1 != sampler2:
                frmt_str = '{} vs. {}: p={}, mean_A,std_A={},{}; mean_B,std_B={},{}'
                print(frmt_str.format(
                    sampler1, sampler2,  bootstrap_test(metrics[sampler1], metrics[sampler2]),
                    np.mean(metrics[sampler1]), np.std(metrics[sampler1]), np.mean(metrics[sampler2]), 
                    np.std(metrics[sampler2])
                ))

    table = AsciiTable(table_data)
    print(table.table)