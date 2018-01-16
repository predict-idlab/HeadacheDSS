import json
from rdflib import URIRef, BNode, Literal, Graph, Namespace
from rdflib.namespace import RDF, FOAF, OWL, RDFS, NamespaceManager
from rdflib.extras.infixowl import Restriction, Individual
import pandas as pd
import numpy as np
from urllib.request import urlopen, quote
import SemanticProcessor.mappings as mappings
import SemanticProcessor.concepts as concepts
#from SemanticProcessor.snomed import getDescriptionsByString

chronicals = Namespace(concepts.BASE_URL)
namespace_manager = NamespaceManager(Graph())

def add_sample(g, id, symptoms, diagnose, prev_attacks, characterisation, severity, location, duration):
	# Creates an URI with its ID and says it's from type chron:Headache
	headache = URIRef(concepts.BASE_URL+'headache#'+str(id))
	g.add( (headache, RDF.type, concepts.headache_type) )

	# For each symptom, we create another triple
	for symptom in symptoms:
		g.add( (headache, concepts.symptom_predicate, mappings.symptom_to_URI[symptom]) )

	# Triple for our diagnose
	g.add( (headache, concepts.diagnose_predicate, mappings.diagnoses_to_URI[diagnose]) )

	# Add triple with previous attacks
	g.add( (headache, concepts.prev_attacks_predicate, Literal(prev_attacks) ) )

	# Add triple with characterisation of pain
	g.add( (headache, concepts.characterisation_predicate, mappings.characterisation_to_URI[characterisation] ) )

	# Add triple with intensity of pain
	g.add( (headache, concepts.intensity_predicate, mappings.severity_to_URI[severity] ) )

	# Add triple for location of pain
	g.add( (headache, concepts.location_predicate, mappings.location_to_URI[location] ) )

	# Add final triple with duration of attack
	g.add( (headache, concepts.duration_predicate, mappings.duration_to_URI[duration]) ) 

def encode(headache_csv, output_path='data/headache_KG.ttl'):
	# Initialize empty triple graph 
	g = Graph()
	g.bind('chron', chronicals, override=True)

	for symptom in mappings.symptom_to_URI:
		query = mappings.symptom_to_URI[symptom].split('/')[-1].replace('_', ' ')
		g.add( (mappings.symptom_to_URI[symptom], RDF.type, concepts.symptom_type) )
		#g.add( (mappings.symptom_to_URI[symptom], OWL.sameAs, URIRef(getDescriptionsByString(query, semTag='finding'))) )

	# For each diagnose, we create a entity
	for diagnose in mappings.diagnoses_to_URI:
		g.add( (mappings.diagnoses_to_URI[diagnose], RDF.type, concepts.diagnose_type) )
		#g.add( (mappings.diagnoses_to_URI[diagnose], OWL.sameAs, URIRef(getDescriptionsByString(mappings.diagnoses_to_URI[diagnose].split('/')[-1], semTag='disorder'))) )

	# Entities for pain characterisations
	for characterisation in mappings.characterisation_to_URI:
		g.add( (mappings.characterisation_to_URI[characterisation], RDF.type, concepts.characterisation_type) )

	# Entities for the location of the pain
	for location in mappings.location_to_URI:
		g.add( (mappings.location_to_URI[location], RDF.type, concepts.location_type) )

	# Entities for the pain severity
	for severity in mappings.severity_to_URI:
		g.add( (mappings.severity_to_URI[severity], RDF.type, concepts.severity_type) )

	duration_bounds = {
		'A': (0, 4),
		'B': (5, 119),
		'C': (120, 239),
		'D': (240, 899),
		'E': (900, 1799),
		'F': (1800, 10799),
		'G': (10800, 14399),
		'H': (14400, 259199),
		'I': (259200, 604799),
		'J': (604800, 'INF'),
	}

	# For each durationGroup, we add a node to our KG with edges to 
	# the corresponding upper and lower bounds
	for duration in mappings.duration_to_URI:
		g.add( (mappings.duration_to_URI[duration], concepts.lb_predicate, Literal(duration_bounds[duration][0])) )
		g.add( (mappings.duration_to_URI[duration], concepts.ub_predicate, Literal(duration_bounds[duration][1])) )
		g.add( (mappings.duration_to_URI[duration], RDF.type, concepts.duration_group_type) )

	# Defining our predicates/properties
	g.add( (concepts.diagnose_predicate, RDF.type, RDF.Property) )
	g.add( (concepts.prev_attacks_predicate, RDF.type, RDF.Property) )
	g.add( (concepts.characterisation_predicate, RDF.type, RDF.Property) )
	g.add( (concepts.intensity_predicate, RDF.type, RDF.Property) )
	g.add( (concepts.location_predicate, RDF.type, RDF.Property) )
	g.add( (concepts.duration_predicate, RDF.type, RDF.Property) )


	# Read the CSV file and iterate over the rows
	for i, row in pd.read_csv(headache_csv).iterrows():

		symptoms = []
		for symptom in mappings.symptoms:
			if symptom in mappings.wrongly_written_symptoms:
				if row[mappings.wrongly_written_symptoms[symptom]] == 'yes':
					symptoms.append(symptom)
			else:
				if row[symptom] == 'yes':
					symptoms.append(symptom)

		add_sample(g, i, symptoms, row['CLASS'], row['previous_attacks'], row['characterisation'],
			       row['severity'], row['location'], row['durationGroup'])

	g.serialize(destination=output_path, format='turtle')

	return g