import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from terminaltables import AsciiTable
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
import click

def bootstrap_test(samples_A, samples_B, repeat=100000, plot=False):
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

# Create the ground truth
migbase = pd.read_csv('data/migbase.csv')
CLASS_MAPPING = {'cluster': 0, 'tension': 1, 'migraine': 2}
GROUND_TRUTH = migbase['CLASS'].map(CLASS_MAPPING)

def calculate_metrics(prediction_file, metric):
    # Read in prediction file, which contains probabilities for all
    # classes. Take argmax as prediction. Calculate accuracy or kappa
    pred_df = pd.read_csv(prediction_file)
    pred_df = pred_df.drop('Unnamed: 0', axis=1)
    predicted_classes = np.argmax(pred_df.values, axis=1)
    if metric == 'accuracy':
        return accuracy_score(GROUND_TRUTH, predicted_classes)
    elif metric == 'kappa':
        return cohen_kappa_score(GROUND_TRUTH, predicted_classes)
    elif metric == 'f1_cluster':
        return f1_score(GROUND_TRUTH, predicted_classes, labels=[0], average='micro')
    elif metric == 'f1_tension':
        return f1_score(GROUND_TRUTH, predicted_classes, labels=[1], average='micro')
    elif metric == 'f1_migraine':
        return f1_score(GROUND_TRUTH, predicted_classes, labels=[2], average='micro')
    elif metric == 'sensitivity_cluster':
        # Sensitivity is equal to TP/(TP+FN) (recall)
        GROUND_TRUTH_BINARY = GROUND_TRUTH.map({0: 1, 1: 0, 2: 0})
        predicted_classes_binary = pd.Series(predicted_classes).map({0: 1, 1: 0, 2: 0})
        tn, fp, fn, tp = confusion_matrix(GROUND_TRUTH_BINARY, predicted_classes_binary).ravel()
        return tp / (fn + tp)
    elif metric == 'specificity_cluster':
        # Specificity is equal to TN/(TN+FP)
        GROUND_TRUTH_BINARY = GROUND_TRUTH.map({0: 1, 1: 0, 2: 0})
        predicted_classes_binary = pd.Series(predicted_classes).map({0: 1, 1: 0, 2: 0})
        tn, fp, fn, tp = confusion_matrix(GROUND_TRUTH_BINARY, predicted_classes_binary).ravel()
        return tn / (tn + fp)
    elif metric == 'sensitivity_tension':
        GROUND_TRUTH_BINARY = GROUND_TRUTH.map({0: 0, 1: 1, 2: 0})
        predicted_classes_binary = pd.Series(predicted_classes).map({0: 0, 1: 1, 2: 0})
        tn, fp, fn, tp = confusion_matrix(GROUND_TRUTH_BINARY, predicted_classes_binary).ravel()
        return tp / (fn + tp)
    elif metric == 'specificity_tension':
        GROUND_TRUTH_BINARY = GROUND_TRUTH.map({0: 0, 1: 1, 2: 0})
        predicted_classes_binary = pd.Series(predicted_classes).map({0: 0, 1: 1, 2: 0})
        tn, fp, fn, tp = confusion_matrix(GROUND_TRUTH_BINARY, predicted_classes_binary).ravel()
        return tn / (tn + fp)
    elif metric == 'sensitivity_migraine':
        GROUND_TRUTH_BINARY = GROUND_TRUTH.map({0: 0, 1: 0, 2: 1})
        predicted_classes_binary = pd.Series(predicted_classes).map({0: 0, 1: 0, 2: 1})
        tn, fp, fn, tp = confusion_matrix(GROUND_TRUTH_BINARY, predicted_classes_binary).ravel()
        return tp / (fn + tp)
    elif metric == 'specificity_migraine':
        GROUND_TRUTH_BINARY = GROUND_TRUTH.map({0: 0, 1: 0, 2: 1})
        predicted_classes_binary = pd.Series(predicted_classes).map({0: 0, 1: 0, 2: 1})
        tn, fp, fn, tp = confusion_matrix(GROUND_TRUTH_BINARY, predicted_classes_binary).ravel()
        return tn / (tn + fp)

def generate_table_data(mode, metric):
    root = 'output/'+mode+'/'
    metrics = {}
    for algorithm in os.listdir(root):
        metrics[algorithm] = []
        for _file in os.listdir(root+algorithm):
            metrics[algorithm].append(calculate_metrics(root+algorithm+os.sep+_file, metric))

    metric_table_data = [['', metric]]
    all_samplers = list(metrics.keys())
    significance_table_data = [[''] + all_samplers]
    for sampler1 in all_samplers:
        metric_table_row = [sampler1, '{}+/-{}'.format(np.round(np.mean(metrics[sampler1]), 6),
                                                np.round(np.std(metrics[sampler1]), 4))]
        metric_table_data.append(metric_table_row)
        significance_table_row = [sampler1]
        for sampler2 in all_samplers:
            if sampler1 != sampler2:
                p_value = bootstrap_test(metrics[sampler1], metrics[sampler2])
                if p_value >= 0.05:
                    significance_table_row.append('  ')
                else:
                    if np.mean(metrics[sampler1]) > np.mean(metrics[sampler2]):
                        symbol = '+'
                    else:
                        symbol = '-'
                    significance_table_row.append(symbol+symbol*(p_value <= 0.01))
            else:
                significance_table_row.append('\\')
        significance_table_data.append(significance_table_row)

    return metric_table_data, significance_table_data

@click.command()
@click.option('--mode', default='oversampling', help='The mode to use: either "oversampling", "features" or "both"')
@click.option('--metric', default='all',
              help='The metrics to calculate: either "all" or on from ["accuracy", "kappa", "f1_cluster", "f1_tension", "f1_migraine"]')
def generate_tables(mode, metric):
    # Depending on the mode, iterate over directories in output/
    # Calculate accuracy or kappa metrics, add them to a dict
    # and apply bootstrap testing
    metrics = ['sensitivity_cluster', 'accuracy', 'kappa', 'f1_cluster', 'f1_tension', 'f1_migraine', 
               'specificity_cluster', 'sensitivity_tension', 'specificity_tension', 'sensitivity_migraine',
               'specificity_migraine']
    if mode == 'both':
        pass
    else:
        if metric == 'all':
            for metric in metrics:
                metric_table_data, significance_table_data = generate_table_data(mode, metric)
                metric_table = AsciiTable(metric_table_data, '{}: {}'.format(mode, metric))
                print(metric_table.table)
                significance_table = AsciiTable(significance_table_data, '{}: {} Significance'.format(mode, metric))
                print(significance_table.table)
        else:
            metric_table_data, significance_table_data = generate_table_data(mode, metric)
            print('{} table'.format(metric))
            metric_table = AsciiTable(metric_table_data, '{}: {}'.format(mode, metric))
            print(metric_table.table)
            significance_table = AsciiTable(significance_table_data, '{}: {} Significance'.format(mode, metric))
            print(significance_table.table)

if __name__ == '__main__':
    generate_tables()
