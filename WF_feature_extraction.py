import rdflib
import copy

# ################################## #
#         CLASSES FOR GRAPHS         #
# ################################## #

class Vertex(object):
    # Keep a (global) counter to 
    # give each node a unique id
    vertex_counter = 0
    
    def __init__(self, name):
        self.reachable = []
        self.name = name
        self.id = Vertex.vertex_counter
        self.previous_name = None
        Vertex.vertex_counter += 1
      

class Graph(object):
    def __init__(self):
        self.vertices = []
        # Transition matrix is a dict of dict, we can
        # access all possible transitions from a vertex
        # by indexing the transition matrix first and then
        # check whether the destination in the dict is True
        self.transition_matrix = {}
        
    def add_vertex(self, vertex):
        """Add vertex to graph and update the 
        transition matrix accordingly"""
        transition_row = {}
        for v in self.vertices:
            transition_row[v] = 0
            self.transition_matrix[v][vertex] = 0
        self.transition_matrix[vertex] = transition_row
        self.vertices.append(vertex)

    def add_edge(self, v1, v2):
        self.transition_matrix[v1][v2] = 1
        
    def remove_edge(self, v1, v2):
        self.transition_matrix[v1][v2] = 0

    def get_neighbors(self, vertex):
        return [k for (k, v) in self.transition_matrix[vertex].items() if v == 1]

    def relabel_nodes(self, mapping):
        for v in self.vertices:
            if v in mapping:
                v.previous_name = v.name
                v.name = mapping[v]


# ################################## #
#             RDF PARSING            #
# ################################## #

namespaces = {'http://chronicals.ugent.be/': 'chron:',
              'http://purl.bioontology.org/ontology/SNOMEDCT/': 'snomed:',
              'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdfs:',
              'http://dbpedia.org/resource/': 'db:',
              'http://semanticweb.org/id/': 'sw:',
              'http://data.semanticweb.org/': 'sw:',
              'http://xmlns.com/foaf/0.1/': 'foaf:',
              'http://swrc.ontoware.org/ontology#': 'swrc:'}


def rdf_to_str(x):
    """Get the string representation for the rdflib object
    and replace urls by keywords defined in `namespaces`.
    Args:
        x (rdflib object)
    Returns:
        x (string)
    """
    x = str(x)
    for ns in namespaces:
        x = x.replace(ns, namespaces[ns])
    return x


def extract_instance(g, inst, d):
    """Extract a (`Graph`) subgraph from the large `rdflib.Graph`
    Args:
        g (rdflib.Graph)    : the large RDF graph
        inst (rdflib.URIRef): the new root of the subgraph
    Returns:
        a `Graph` object rooted at `inst`
    """
    subgraph = Graph()
    # Add the instance with custom label (root)
    root = Vertex('root')
    subgraph.add_vertex(root)

    nodes_to_explore = set([(inst, root)])
    for i in range(d):
        if len(nodes_to_explore):
            # Convert set to list, since we cannot change size
            for rdf, v in list(nodes_to_explore):  
                # Create a SPARQL query to filter out 
                # all triples with subject = `rdf`
                qres = g.query("""SELECT ?p ?o WHERE {
                                    ?s ?p ?o .
                               }""",
                               initBindings={'s': rdf})
                for row in qres:
                    # Add two new nodes
                    v1 = Vertex(rdf_to_str(row[0]))
                    v2 = Vertex(rdf_to_str(row[1]))
                    subgraph.add_vertex(v1)
                    subgraph.add_vertex(v2)
                    
                    # And two new edges
                    if rdf == inst:
                        subgraph.add_edge(root, v1)
                    else:
                        subgraph.add_edge(v, v1)
                    subgraph.add_edge(v1, v2)
                    
                    # Add the object as new node to explore
                    nodes_to_explore.add((row[1], v2))
                    
                # Remove the node we just explored
                nodes_to_explore.remove((rdf, v))
    return subgraph


# ################################## #
#      Weisfeiler-Lehman kernel      #
# ################################## #


def wf_relabel_graph(g, s_n_to_counter, n_iterations=5, verbose=False):
    """Weisfeiler-Lehman relabeling algorithm, used to calculate the
    corresponding kernel.
    Args:
        g (`Graph`): the knowledge graph, mostly first extracted
                     from a larger graph using `extract_instance`
        s_n_to_counter (dict): global mapping function that maps a 
                               multi-set label to a unique integer
        n_iterations (int): maximum subtree depth
    Returns:
        label_mappings (dict): for every subtree depth (or iteration),
                               the mapping of labels is stored
                               (key = old, value = new)
    """
    
    # Our resulting label function (a map for each iterations)
    label_mappings = {}
    multi_set_labels = {}
    
    # Take a deep copy of our graph, since we are going to relabel its nodes
    g = copy.deepcopy(g)

    for n in range(n_iterations):
        labels = {}
        multi_set_labels[n] = {}
        for vertex in g.vertices:
            # First we create multi-set labels, s_n composed as follow:
            # Prefix = label of node 
            # Suffix = sorted labels of neighbors (reachable by node)
            # If n == 0, we just use the name of the vertex
            if n == 0:
                s_n = vertex.name
            else:
                # g.edges(v) returns all edges coming from v
                s_n = '-'.join(sorted(set(map(str, [x.name for x in g.get_neighbors(vertex)]))))
            
            multi_set_labels[n][vertex.id] = s_n

            if n > 0 and multi_set_labels[n-1][vertex.id] != s_n:
                s_n = (str(vertex.name) + '-' + s_n).rstrip('-')
            elif n > 0:
                s_n = (str(vertex.previous_name) + '-' + str(multi_set_labels[n-1][vertex.id])).rstrip('-')
                vertex.name = vertex.previous_name
                
            labels[vertex] = s_n
                
        # We now map each unique label s_n to an integer
        for s_n in sorted(set(labels.values())):
            if s_n not in s_n_to_counter:
                s_n_to_counter[s_n] = len(s_n_to_counter)

                # If n == 0, then the label is an rdf identifier,
                # we map it first to an integer and store that integer
                # as mapping as well
                if n == 0:
                    s_n_to_counter[str(s_n_to_counter[s_n])] = s_n_to_counter[s_n]
        
        # Construct a dict that maps a node to a new integer,
        # based on its multi-set label
        label_mapping = {}
        for vertex in g.vertices:
            # Get its multi-set label
            s_n = labels[vertex]
            
            # Get the integer corresponding to this multi-set label
            label_mapping[vertex] = s_n_to_counter[s_n]
            
        # Append the created label_mapping to our result dict
        label_mappings[n] = label_mapping
        
        if verbose:
            print('Iteration {}:'.format(n))
            print('-'*25)
            for node in label_mappings[n]:
                print(node.id, node.name, labels[node], '-->', label_mappings[n][node])
            print('\n'*2)

        # Relabel the nodes in our graph, ready for the next iteration
        g.relabel_nodes(label_mapping)
        
    return label_mappings

def wf_kernel(g, inst1, inst2, n_iterations=8, verbose=False):
    # First we extract subgraphs, rooted at the given instances from 
    # our larger knowledge graph, with a maximum depth of `n_iterations`
    g1 = extract_instance(g, inst1, n_iterations//2)
    g2 = extract_instance(g, inst2, n_iterations//2)
    
    # The global mapping function that maps multi-set labels to integers
    # used for both graphs
    s_n_to_counter = {}
    
    # Weisfeiler-Lehman relabeling
    g1_label_function = wf_relabel_graph(g1, s_n_to_counter, n_iterations=n_iterations, verbose=verbose)
    g2_label_function = wf_relabel_graph(g2, s_n_to_counter, n_iterations=n_iterations, verbose=verbose)

    # Iterate over the different iterations and count the number of similar labels between
    # the graphs. Make sure we do not count labels double by first removing all labels
    # from the previous iteration.
    values = []
    for n in range(n_iterations):
        g1_labels = set(g1_label_function[n].values())
        g2_labels = set(g2_label_function[n].values())
        if n == 0:
            values.append(len(g1_labels.intersection(g2_labels)))
        else:
            prev_g1_labels = set(g1_label_function[n-1].values())
            prev_g2_labels = set(g2_label_function[n-1].values())
            values.append(len((g1_labels - prev_g1_labels).intersection(g2_labels - prev_g2_labels)))
            
    return values

# ################################## #
#         Generate results           #
# ################################## #   

import numpy as np
import pandas as pd
from scipy import spatial

from SemanticProcessor.decoder import decode
from SemanticProcessor import encoder, generator

from sklearn.metrics import f1_score, accuracy_score, classification_report, pairwise, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

import pickle

import itertools

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