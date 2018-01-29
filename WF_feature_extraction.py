import pickle
import itertools
import numpy as np
import pandas as pd
import rdflib
import tqdm

from SemanticProcessor.decoder import decode
from SemanticProcessor import encoder, generator
from WFL.kernel import wf_kernel
from sklearn.metrics import accuracy_score, f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')


# ################################## #
#    Load our headache & ICHD KG     #
# ################################## # 
print("Creating the migbase + ICHD knowledge graph (rdflib)...")
g = rdflib.Graph()
g.parse('data/headache_KG.ttl', format='turtle')

# First, we build a dictionary with labels
labels = {}
qres = g.query("""SELECT ?headache ?label WHERE {
                    ?headache chron:isType ?label .
               }""",
               initNs={'chron': rdflib.Namespace('http://chronicals.ugent.be/')})
for row in qres:
    labels[row[0]] = row[1]

# Then, we remove all triples that have chron:isType as predicate
# since these form a data leak...
new_g = rdflib.Graph()
qres = g.query("""SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                    MINUS {
                        ?s chron:isType ?o .
                    }
               }""",
               initNs={'chron': rdflib.Namespace('http://chronicals.ugent.be/')})
for s, p, o in qres:
    if 'ugent' in str(p) and 'isType' in str(p): 
        # This shouldn't happen... (if the query works)
        print('We added the label to the graph...')
    new_g.add((s, p, o))

g = new_g

# Create a 'prototype' KG for each class based on the ICHD knowledge base
ichd_kg = rdflib.Graph()
ichd_kg.parse('data/ICHD_KB.ttl', format='turtle')


qres = ichd_kg.query("""SELECT ?diagnose ?property ?item WHERE {
                          ?diagnose rdfs:subClassOf ?bnode1 .
                          ?bnode1 owl:intersectionOf ?bnode2 .
                          ?bnode2 rdf:type owl:Restriction .
                          ?bnode2 owl:onProperty ?property .
                          ?bnode2 (owl:oneValueFrom|owl:someValuesFrom)/rdf:rest*/rdf:first ?item .
                       }
                     """,
                     initNs={'rdf':  rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
                                'rdfs': rdflib.Namespace('http://www.w3.org/2000/01/rdf-schema#'),
                                 'owl':  rdflib.Namespace('http://www.w3.org/2002/07/owl#'),
                                 'chron':  rdflib.Namespace('http://chronicals.ugent.be/')})

# The roots of the prototype subgraphs
prototypes = [rdflib.URIRef('http://chronicals.ugent.be/Cluster'),
              rdflib.URIRef('http://chronicals.ugent.be/Tension'),
              rdflib.URIRef('http://chronicals.ugent.be/Migraine')]

for s,p,o in qres:
    if 'ugent' in str(p) and 'isType' in str(p): print('We added the label to the graph...')
    g.add((s,p,o))

# Convert URIRefs to integers to use sklearn metrics
uri_to_int = {rdflib.URIRef('http://chronicals.ugent.be/Cluster'): 0,
              rdflib.URIRef('http://chronicals.ugent.be/Tension'): 1,
              rdflib.URIRef('http://chronicals.ugent.be/Migraine'): 2}


# ################################## #
#       Create feature vectors       #
# ################################## # 

print('Generating distances from each sample (represented as a graph) to each class concept (graph)...')
correct = 0
total = 0
real_labels = []
predicted_labels = []
wf_features = {}
for headache in tqdm.tqdm(labels.keys()):
    feature_vector = [sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes]
    wf_features[int(str(headache).split('#')[-1])] = feature_vector
    correct += prototypes[np.argmax([sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes])] == labels[headache]
    total += 1
    real_labels.append(uri_to_int[labels[headache]])
    predicted_labels.append(uri_to_int[prototypes[np.argmax([sum(wf_kernel(g, prototype, headache)[1:]) for prototype in prototypes])]])

print('Metrics...')
print('Unsupervised accuracy:', accuracy_score(real_labels, predicted_labels))
print('Unsupervised F1:', f1_score(real_labels, predicted_labels, average='micro'))
print(classification_report(real_labels, predicted_labels))
print('Writing features to data/...')
pickle.dump(wf_features, open('data/wf_features.p', 'wb'))