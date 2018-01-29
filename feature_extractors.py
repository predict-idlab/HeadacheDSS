import pickle
import itertools

import numpy as np
import pandas as pd

from scipy import spatial
from sklearn.metrics import f1_score, accuracy_score, classification_report, pairwise, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn import tree

from SemanticProcessor.decoder import decode
from SemanticProcessor import encoder, generator

import warnings
warnings.filterwarnings('ignore')



# # ################################## #
# #    Load our headache & ICHD KG     #
# # ################################## # 

# g = rdflib.Graph()
# g.parse('SemanticProcessor/data/headache_KG.ttl', format='turtle')

# # First, we build a dictionary with labels
# labels = {}
# qres = g.query("""SELECT ?headache ?label WHERE {
#                     ?headache chron:isType ?label .
#                }""",
#                initNs={'chron': rdflib.Namespace('http://chronicals.ugent.be/')})
# for row in qres:
#     labels[row[0]] = row[1]

# # Then, we remove all triples that have chron:isType as predicate
# # since these form a data leak...
# new_g = rdflib.Graph()
# qres = g.query("""SELECT ?s ?p ?o WHERE {
#                     ?s ?p ?o .
#                     MINUS {
#                         ?s chron:isType ?o .
#                     }
#                }""",
#                initNs={'chron': rdflib.Namespace('http://chronicals.ugent.be/')})
# for s, p, o in qres:
#     if 'ugent' in str(p) and 'isType' in str(p): print('We added the label to the graph...')
#     new_g.add((s, p, o))

# g = new_g

# # Create a 'prototype' KG for each class based on the ICHD knowledge base
# ichd_kg = rdflib.Graph()
# ichd_kg.parse('SemanticProcessor/data/ICHD_KB.ttl', format='turtle')


# qres = ichd_kg.query("""SELECT ?diagnose ?property ?item WHERE {
#                           ?diagnose rdfs:subClassOf ?bnode1 .
#                           ?bnode1 owl:intersectionOf ?bnode2 .
#                           ?bnode2 rdf:type owl:Restriction .
#                           ?bnode2 owl:onProperty ?property .
#                           ?bnode2 (owl:oneValueFrom|owl:someValuesFrom)/rdf:rest*/rdf:first ?item .
#                        }
#                      """,
#                      initNs={'rdf':  rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
#                                 'rdfs': rdflib.Namespace('http://www.w3.org/2000/01/rdf-schema#'),
#                                  'owl':  rdflib.Namespace('http://www.w3.org/2002/07/owl#'),
#                                  'chron':  rdflib.Namespace('http://chronicals.ugent.be/')})

# # The roots of the prototype subgraphs
# prototypes = [rdflib.URIRef('http://chronicals.ugent.be/Cluster'),
#               rdflib.URIRef('http://chronicals.ugent.be/Tension'),
#               rdflib.URIRef('http://chronicals.ugent.be/Migraine')]

# for s,p,o in qres:
#     if 'ugent' in str(p) and 'isType' in str(p): print('We added the label to the graph...')
#     g.add((s,p,o))

# # Convert URIRefs to integers to use sklearn metrics
# uri_to_int = {rdflib.URIRef('http://chronicals.ugent.be/Cluster'): 0,
#               rdflib.URIRef('http://chronicals.ugent.be/Tension'): 1,
#               rdflib.URIRef('http://chronicals.ugent.be/Migraine'): 2}


# # ################################## #
# #       Create feature vectors       #
# # ################################## # 

# correct = 0
# total = 0
# real_labels = []
# predicted_labels = []
# wf_features = {}
# for headache in labels.keys():
#     feature_vector = [sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes]
#     wf_features[int(str(headache).split('#')[-1])] = feature_vector
#     correct += prototypes[np.argmax([sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes])] == labels[headache]
#     total += 1
#     print('Prediction:', prototypes[np.argmax([sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes])],
#           '|| Real:', labels[headache],
#           ' || Total:', total,
#           ' || Accuracy:', correct/total)
#     real_labels.append(uri_to_int[labels[headache]])
#     predicted_labels.append(uri_to_int[prototypes[np.argmax([sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes])]])

# print('Unsupervised accuracy:', accuracy_score(real_labels, predicted_labels))
# print('Unsupervised F1:', f1_score(real_labels, predicted_labels, average='micro'))
# print(classification_report(real_labels, predicted_labels))
# pickle.dump(wf_features, open('wf_features.p', 'wb'))


# ################################## #
#       Load our original data       #
# ################################## # 

wf_features = pickle.load(open('wf_features.p', 'rb'))
migbase = pd.read_csv('SemanticProcessor/data/migbase.csv')
#migbase = migbase.sample(50)
_columns = []
for col in migbase.columns:
    if len(np.unique(migbase[col])) > 1: 
        _columns.append(col)
chronicals = pd.read_csv('SemanticProcessor/data/chronicals_features.csv')
migbase = migbase[migbase['CLASS'] != 'no headache']

migbase = migbase[_columns]
chronicals = chronicals[_columns]

def preprocess(features, labels):
    mapping_per_class = {

        'durationGroup': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                          'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9},
        'severity': {'mild': 0, 'moderate': 1, 'severe': 2, 'very severe': 3},
        'previous_attacks': {'2-4': 0, '5-9': 1, '10-19': 2, '20': 3},

        # One-hot-encoding performs worse than integer encoding
        # for these non-ordinal categorical variables
        'location': {'unilateral': 0, 'bilateral': 1, 'orbital': 2},
        'characterisation': {'stabbing': 0, 'pressing': 1, 'pulsating': 2}
    }
    for col in features.columns:
        unique_values = np.unique(features[col])
        if 'no' in unique_values or 'yes' in unique_values:
            features[col] = features[col].map({'no': 0, 'yes': 1})
            mapping_per_class[col] = {'no': 0, 'yes': 1}
        else:
            features[col] = features[col].map(mapping_per_class[col])
    return features, labels.map( {'cluster': 0, 'tension': 1, 'migraine': 2} )

features = migbase.drop('CLASS', axis=1)
labels = migbase['CLASS']

features, labels = preprocess(features, labels)

chronicals_features = chronicals.drop('CLASS', axis=1)
chronicals_labels = chronicals['CLASS']

chronicals_features, chronicals_labels = preprocess(chronicals_features, chronicals_labels)


def wf_similarity(X, Y):
    # Y is unused (hardcoded below), but still included as parameter to have same syntax as
    # sklearn pairwise metrics
    similarities = np.zeros((len(X), 3))
    for i, (idx, row) in enumerate(X.iterrows()):
        similarities[i] = [wf_features[idx][0], wf_features[idx][1], wf_features[idx][2]]
    return similarities


def add_similarities(X_train, y_train, X_test, y_test):
    # All the similarities, stored as tuples:
    # (name, callback, maximize)
    similarities = [
        ('WF', wf_similarity, 1),
        ('Chi2', pairwise.chi2_kernel, 1),
        ('Laplace', pairwise.laplacian_kernel, 1),
        ('Cos', pairwise.cosine_similarity, 1),
        ('RBF', pairwise.rbf_kernel, 1),
    ]

    def get_distance_columns(x): 
        return [str(x)+'_cluster', str(x)+'_tension', str(x)+'_migraine']

    cluster_prototype = X_train.loc[(y_train[y_train==0]).index].mean()
    tension_prototype = X_train.loc[(y_train[y_train==1]).index].mean()
    migraine_prototype = X_train.loc[(y_train[y_train==2]).index].mean()

    train_features = X_train.copy()
    test_features = X_test.copy()
    for name, method, maximize in similarities:
        train_distances = method(train_features, [cluster_prototype, tension_prototype, migraine_prototype])
        train_closest = [np.argmin, np.argmax][maximize](train_distances, axis=1)
        test_distances = method(test_features, [cluster_prototype, tension_prototype, migraine_prototype])
        test_closest = [np.argmin, np.argmax][maximize](test_distances, axis=1)

        distance_columns = get_distance_columns(name)
        for col in distance_columns:
            X_train[col] = np.NaN
            X_test[col] = np.NaN

        X_train[distance_columns] = train_distances
        X_train[name+'_closest'] = train_closest
        X_test[distance_columns] = test_distances
        X_test[name+'_closest'] = test_closest

acc_scores = {}
f1_scores = {}
cohen_scores = {}
for _ in range(100):
    SEED = np.random.randint(1000000)
    skf = StratifiedKFold(n_splits=5, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train = features.iloc[train_idx, :].copy()
        X_test = features.iloc[test_idx, :].copy()

        y_train = labels.iloc[train_idx].copy()
        y_test = labels.iloc[test_idx].copy()

        dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=SEED)
        dt.fit(X_train, y_train)
        acc, f1 = accuracy_score(y_test, dt.predict(X_test)), f1_score(y_test, dt.predict(X_test), average='weighted')
        kappa = cohen_kappa_score(y_test, dt.predict(X_test))
        print('Fold {}, Original: Accuracy = {}, F1 = {}'.format(fold + 1, acc, f1))
        if 'original' not in acc_scores:
            acc_scores['original'] = [acc]
            f1_scores['original'] = [f1]
            cohen_scores['original'] = [kappa]
        else:
            acc_scores['original'].append(acc)
            f1_scores['original'].append(f1)
            cohen_scores['original'].append(kappa)

        original_features = list(X_train.columns)
        add_similarities(X_train, y_train, X_test, y_test)
        for col in X_test.columns:
            if 'closest' in col:
                acc, f1 = accuracy_score(y_test, X_test[col]), f1_score(y_test, X_test[col], average='weighted')
                kappa = cohen_kappa_score(y_test, X_test[col])
                print('Only', col.split('-')[0], acc, f1)
                if 'only '+col.split('-')[0] not in acc_scores:
                    acc_scores['only '+col.split('-')[0]] = [acc]
                    f1_scores['only '+col.split('-')[0]] = [f1]
                    cohen_scores['only '+col.split('-')[0]] = [kappa]
                else:
                    acc_scores['only '+col.split('-')[0]].append(acc)
                    f1_scores['only '+col.split('-')[0]].append(f1)
                    cohen_scores['only '+col.split('-')[0]].append(kappa)


        similarities = ['WF', 'Chi2', 'Laplace', 'Cos', 'RBF']
        for sim in similarities:
            dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=SEED)
            dt.fit(X_train[original_features + [sim+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]], y_train)
            preds = dt.predict(X_test[original_features + [sim+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]])
            acc, f1 = accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted')
            kappa = cohen_kappa_score(y_test, preds)
            print('Fold {}, Original + {}: Accuracy = {}, F1 = {}'.format(fold + 1, sim, acc, f1))
            if sim not in acc_scores:
                acc_scores[sim] = [acc]
                f1_scores[sim] = [f1]
                cohen_scores[sim] = [kappa]
            else:
                acc_scores[sim].append(acc)
                f1_scores[sim].append(f1)
                cohen_scores[sim].append(kappa)

        for sim1, sim2 in itertools.combinations(similarities, 2):
            dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=SEED)
            dt.fit(X_train[original_features + [sim1+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]
                           + [sim2+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]], y_train)
            preds = dt.predict(X_test[original_features + [sim1+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]
                               + [sim2+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]])
            acc, f1 = accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted')
            kappa = cohen_kappa_score(y_test, preds)
            print('Fold {}, Original + {} + {}: Accuracy = {}, F1 = {}'.format(fold + 1, sim1, sim2, acc, f1))
            if (sim1, sim2) not in acc_scores:
                acc_scores[(sim1, sim2)] = [acc]
                f1_scores[(sim1, sim2)] = [f1]
                cohen_scores[(sim1, sim2)] = [kappa]
            else:
                acc_scores[(sim1, sim2)].append(acc)
                f1_scores[(sim1, sim2)].append(f1)
                cohen_scores[(sim1, sim2)].append(kappa)

        print('-'*40)


pickle.dump(acc_scores, open('acc_per_feature.p', 'wb'))
pickle.dump(f1_scores, open('f1_per_feature.p', 'wb'))
pickle.dump(cohen_scores, open('cohen_per_feature.p', 'wb'))


# dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=1337)
# dt.fit(features, labels)
# print('Chronicals: Original: Accuracy = {}, F1 = {}'.format(accuracy_score(chronicals_labels, dt.predict(chronicals_features)),
#                                                             f1_score(chronicals_labels, dt.predict(chronicals_features), average='weighted')))


# n_cluster_samples = len(labels[labels == 2]) - len(labels[labels == 0])
# n_tension_samples = len(labels[labels == 2]) - len(labels[labels == 1])

# # Generate the samples in the form of RDF-files
# generator.generate_samples('Cluster', ['SemanticProcessor/data/headache_KG.ttl', 'SemanticProcessor/data/ICHD_KB.ttl'], 
#                            n=n_cluster_samples, id_offset=1000,
#                            output_path='SemanticProcessor/data/generated_samples_cluster.ttl')
# generator.generate_samples('Tension', ['SemanticProcessor/data/headache_KG.ttl', 'SemanticProcessor/data/ICHD_KB.ttl'], 
#                            n=n_tension_samples, id_offset=2000,
#                            output_path='SemanticProcessor/data/generated_samples_tension.ttl')

# # Decoded the generated RDF-files to get a pandas dataframe
# new_df = decode(rdflib.Graph().parse("SemanticProcessor/data/generated_samples_cluster.ttl", format="turtle"))

# # Apply the same pre-processing to our new dataframe
# new_features = new_df.drop(['index', 'CLASS'], axis=1)
# new_labels = new_df['CLASS']
# new_features, new_labels = preprocess(new_features, new_labels)
# new_features = new_features.reindex_axis(features.columns, axis=1)

# # Append it
# features = pd.concat([features, new_features])
# labels = pd.concat([labels, new_labels])

# new_df = decode(rdflib.Graph().parse("SemanticProcessor/data/generated_samples_tension.ttl", format="turtle"))

# # Apply the same pre-processing to our new dataframe
# new_features = new_df.drop(['index', 'CLASS'], axis=1)
# new_labels = new_df['CLASS']
# new_features, new_labels = preprocess(new_features, new_labels)
# new_features = new_features.reindex_axis(features.columns, axis=1)

# # Append it
# features = pd.concat([features, new_features])
# labels = pd.concat([labels, new_labels])

# original_features = list(features.columns)
# add_similarities(features, labels, chronicals_features, chronicals_labels)
# for col in X_test.columns:
#     if 'closest' in col:
#         print('Only', col.split('-')[0], accuracy_score(chronicals_labels, chronicals_features[col]))

# similarities = ['WF', 'Chi2', 'Laplace', 'Cos', 'RBF']
# for sim in similarities:
#     dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=1337)
#     dt.fit(features[original_features + [sim+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]], labels)
#     preds = dt.predict(chronicals_features[original_features + [sim+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]])
#     print('Chronicals: Original + {}: Accuracy = {}, F1 = {}'.format(sim, accuracy_score(chronicals_labels, preds),
#                                                                   f1_score(chronicals_labels, preds, average='weighted')))

# for sim1, sim2 in itertools.combinations(similarities, 2):
#     dt = tree.DecisionTreeClassifier(criterion='entropy', random_state=1337)
#     dt.fit(features[original_features + [sim1+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]
#                    + [sim2+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]], labels)
#     preds = dt.predict(chronicals_features[original_features + [sim1+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]
#                        + [sim2+'_'+x for x in ['cluster', 'tension', 'migraine', 'closest']]])
#     print('Chronicals: Original + {} + {}: Accuracy = {}, F1 = {}'.format(sim1, sim2, accuracy_score(chronicals_labels, preds),
#                                                                       f1_score(chronicals_labels, preds, average='weighted')))