import pickle
import itertools
import os

import numpy as np
import pandas as pd
import tqdm
import click

from scipy import spatial
from sklearn.metrics import f1_score, accuracy_score, classification_report, pairwise, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')


def preprocess(features, labels):
    mapping_per_class = {
        # Duration group is a discretization of the headache duration
            # It is thus an ordinal variable.
            'durationGroup': {
                'A': 0,
                    'B': 1,
                    'C': 2,
                    'D': 3,
                    'E': 4,
                    'F': 5,
                    'G': 6,
                    'H': 7,
                    'I': 8,
                    'J': 9
            },

            # Severity is an ordinal variable as well
            'severity': {
                'mild': 0,
                    'moderate': 1,
                    'severe': 2,
                    'very severe': 3
            },

            # The number of previous similar headache attacks is ordinal
            'previous_attacks': {
                '2-4': 0,
                    '5-9': 1,
                    '10-19': 2,
                    '20': 3
            },

            # One-hot-encoding performs worse than integer encoding
            # for these non-ordinal categorical variables
            'location': {
                'unilateral': 0,
                    'bilateral': 1,
                    'orbital': 2
            },

            'characterisation': {
                'stabbing': 0,
                    'pressing': 1,
                    'pulsating': 2
            }
    }
    # If we find a column with only 'yes' and 'no' as values
    # Then we map 'yes' to 1 and 'no' to 0
    # Else, we use a mapping defined above
    for col in features.columns:
        unique_values = np.unique(features[col])
        if 'no' in unique_values or 'yes' in unique_values:
            features[col] = features[col].map({'no': 0, 'yes': 1})
            mapping_per_class[col] = {'no': 0, 'yes': 1}
        else:
            if col in mapping_per_class:
                features[col] = features[col].map(mapping_per_class[col])
            else:
                # If it's not binary, or we did not define a mapping
                # then we do not need the feature
                features = features.drop(col, axis=1)
    return features, labels.map( {'cluster': 0, 'tension': 1, 'migraine': 2} )


def wf_similarity(X, Y):
    # Y is unused (hardcoded below), but still included as parameter to have same syntax as
    # sklearn pairwise metrics
    # wf_features is a dict with key = id and values = list with 3 similarity scores 
    wf_features = pickle.load(open('data/wf_features.p', 'rb'))
    similarities = np.zeros((len(X), 3))
    for i, (idx, row) in enumerate(X.iterrows()):
        similarities[i] = [wf_features[idx][0], wf_features[idx][1], wf_features[idx][2]]
    return similarities


def add_similarities(X_train, y_train, X_test, y_test, similarity):
    """Calculate similarity scores and append them to the
    feature dataframe"""
    _similarities = {
        'WF': wf_similarity,
        'Chi2': pairwise.chi2_kernel,
        'Laplace': pairwise.laplacian_kernel,
        'Cos': pairwise.cosine_similarity,
        'RBF': pairwise.rbf_kernel,
    }

    def get_distance_columns(x): 
        return [str(x)+'_cluster', str(x)+'_tension', str(x)+'_migraine']

    cluster_prototype = X_train.loc[(y_train[y_train==0]).index].mean()
    tension_prototype = X_train.loc[(y_train[y_train==1]).index].mean()
    migraine_prototype = X_train.loc[(y_train[y_train==2]).index].mean()

    train_features = X_train.copy()
    test_features = X_test.copy()
    new_X_train = X_train.copy()
    new_X_test = X_test.copy()

    method = _similarities[similarity]

    train_distances = method(train_features, [cluster_prototype, tension_prototype, migraine_prototype])
    train_closest = np.argmax(train_distances, axis=1)
    test_distances = method(test_features, [cluster_prototype, tension_prototype, migraine_prototype])
    test_closest = np.argmax(test_distances, axis=1)

    distance_columns = get_distance_columns(similarity)
    for col in distance_columns:
        new_X_train[col] = np.NaN
        new_X_test[col] = np.NaN

    new_X_train[distance_columns] = train_distances
    new_X_train[similarity+'_closest'] = train_closest
    new_X_test[distance_columns] = test_distances
    new_X_test[similarity+'_closest'] = test_closest

    return new_X_train, y_train, new_X_test, y_test

@click.command()
@click.option('--n_simulations', default=100, help='The number of simulations')
def run_simulations(n_simulations):
    similarities = ['None', 'WF', 'Chi2', 'Laplace', 'Cos', 'RBF']

    # Create the output directory and subdirectories if needed
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('output' + os.sep + 'features'):
        os.makedirs('output' + os.sep + 'features')
    for sim in similarities:
        if not os.path.exists('output' + os.sep + 'features' + os.sep + sim):
            os.makedirs('output' + os.sep + 'features' + os.sep + sim)

    # Read the migbase data
    migbase = pd.read_csv('data/migbase.csv')

    # Remove single sampmle of no headache class
    migbase = migbase[migbase['CLASS'] != 'no headache']

    # Drop columns with only 1 unique value
    _columns = []
    for col in migbase.columns:
        if len(np.unique(migbase[col])) > 1: 
            _columns.append(col)
    migbase = migbase[_columns]
    
    # Apply pre-processing (map strings to ints)
    features = migbase.drop('CLASS', axis=1)
    labels = migbase['CLASS']
    features, labels = preprocess(features, labels) 


    # Apply cross-validation `n_simulations` times
    for _ in tqdm.tqdm(range(n_simulations)):
        SEED = np.random.randint(1000000)
        # Iterate over the different similarities
        for sim in similarities:
            # Apply 5-fold CV
            preds = np.zeros((len(labels), 3))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
                # Partition in train and test data
                X_train = features.iloc[train_idx, :].copy()
                X_test = features.iloc[test_idx, :].copy()
                y_train = labels.iloc[train_idx].copy()
                y_test = labels.iloc[test_idx].copy()

                if sim == 'None':
                    # Fit the baseline (no extra features)
                    dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=SEED)
                    dt.fit(X_train, y_train)
                    preds[test_idx, :] = dt.predict_proba(X_test)
                else:
                    # Add new features and fit decision tree
                    X_train_aug, y_train_aug, X_test_aug, y_test_aug = add_similarities(X_train, y_train, X_test, y_test, sim)
                    dt.fit(X_train_aug, y_train_aug)
                    preds[test_idx, :] = dt.predict_proba(X_test_aug)

            # Write away predictions
            preds_df = pd.DataFrame(
                preds,
                columns=['cluster_prob',
                         'tension_prob',
                         'migraine_prob']
            )
            preds_df.to_csv('output' + os.sep + 'features' +
                            os.sep + sim + os.sep + 'preds_' + str(SEED) + '.csv')

if __name__ == '__main__':
    run_simulations()