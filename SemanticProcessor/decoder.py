import json
from rdflib import URIRef, BNode, Literal, Graph, Namespace
from rdflib.namespace import RDF, FOAF, OWL, RDFS, NamespaceManager
from rdflib.extras.infixowl import Restriction, Individual
import pandas as pd
import numpy as np
from urllib.request import urlopen, quote
import SemanticProcessor.mappings as mappings
import SemanticProcessor.concepts as concepts

chronicals = Namespace(concepts.BASE_URL)
namespace_manager = NamespaceManager(Graph())

def decode(g):
	query = """SELECT ?headache ?duration ?characterisation ?intensity ?location ?prev_attacks ?diagnosis WHERE {
		?headache rdf:type ?headache_type .
		?headache ?duration_predicate ?duration .
		?headache ?characterisation_predicate ?characterisation . 
		?headache ?intensity_predicate ?intensity .
		?headache ?location_predicate ?location .
		?headache ?prev_atk_predicate ?prev_attacks .
		?headache ?diagnosis_predicate ?diagnosis .
	}"""

	symptom_query = """SELECT ?symptom WHERE{
		?headache ?symptom_predicate ?symptom .
	}
	"""
	
	qres = g.query(query, initNs={'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')},
				   initBindings={'headache_type': concepts.headache_type,
				   				 'duration_predicate': concepts.duration_predicate,
				   				 'characterisation_predicate': concepts.characterisation_predicate,
				   				 'intensity_predicate': concepts.intensity_predicate,
				   				 'location_predicate': concepts.location_predicate,
				   				 'prev_atk_predicate': concepts.prev_attacks_predicate,
				   				 'diagnosis_predicate': concepts.diagnose_predicate})

	symptoms_list = mappings.symptoms
	_columns = ['index', 'CLASS', 'durationGroup', 'location', 'severity', 'characterisation', 'previous_attacks'] + symptoms_list
	vectors = []

	for headache, duration, characterisation, intensity, location, prev_attacks, diagnosis in qres:
		_id = int(headache.split('#')[-1])
		duration = mappings.URI_to_duration[duration]
		characterisation = mappings.URI_to_characterisation[characterisation]
		severity = mappings.URI_to_severity[intensity]
		location = mappings.URI_to_location[location]
		diagnosis = mappings.URI_to_diagnoses[diagnosis]
		
		symptoms = {}
		for symptom in mappings.symptoms:
			symptoms[symptom] = 'no'
		for symptom in g.query(symptom_query, initBindings={'headache': headache, 'symptom_predicate': concepts.symptom_predicate}):
			symptoms[mappings.URI_to_symptom[symptom[0]]] = 'yes'
		
		vector = [_id, diagnosis, duration, location, severity, characterisation, prev_attacks.toPython()]
		for symptom in symptoms_list:
			vector.append(symptoms[symptom])
		vectors.append(vector)

	for i in range(len(_columns)):
		if _columns[i] in mappings.wrongly_written_symptoms:
			_columns[i] = mappings.wrongly_written_symptoms[_columns[i]]

	df = pd.DataFrame(vectors, columns=_columns)
	return df
