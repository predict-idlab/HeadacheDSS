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
PURL_PREFIX = 'http://purl.bioontology.org/ontology/SNOMEDCT/'

def generate_ICHD_KB():
	g = Graph()
	# Some namespace fixing
	g.bind('chron', chronicals, override=True)
	g.bind('owl', Namespace('http://www.w3.org/2002/07/owl#'), override=True)


	# ####################################################### #
	#					CLUSTER headache                      #
	# #########################################################
	cluster = URIRef(concepts.BASE_URL+'Cluster')
	g.add( (cluster, RDF.type, OWL.Class ) )

	# Link to SNOMED (from 'Headche disorder > Trigeminal autonomic cephalalgia')
	g.add( (cluster, OWL.sameAs, URIRef(PURL_PREFIX + '193031009')) )

	# Has one or more of the following symptoms
	blank_node = symptom_list_object = BNode()
	symptom_list = ['conjunctival_injection', 'lacrimation', 'nasal_congestion', 
				    'rhinorrhoea', 'eyelid_oedema', 'sweating', 'miosis', 'ptosis']
	for i, symptom in enumerate(symptom_list):
		g.add( (blank_node, RDF.first, mappings.symptom_to_URI[symptom]) )
		blank_node2 = BNode()
		if i < len(symptom_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Severe or very severe unilateral orbital
	blank_node = severity_list_object = BNode()
	severity_list = ['severe']  # , 'very severe'
	for i, severity in enumerate(severity_list):
		g.add( (blank_node, RDF.first, mappings.severity_to_URI[severity]) )
		blank_node2 = BNode()
		if i < len(severity_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Severe or very severe unilateral orbital
	blank_node = location_list_object = BNode()
	location_list = ['unilateral', 'orbital']
	for i, location in enumerate(location_list):
		g.add( (blank_node, RDF.first, mappings.location_to_URI[location]) )
		blank_node2 = BNode()
		if i < len(location_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Pain lasting 15-180 min (when untreated)
	blank_node = duration_list_object = BNode()
	duration_list = ['E', 'F']
	for i, duration in enumerate(duration_list):
		g.add( (blank_node, RDF.first, mappings.duration_to_URI[duration]) )
		blank_node2 = BNode()
		if i < len(duration_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# TODO: 'a sense of restlessness or agitation' not yet encoded
	# TODO: 'Attacks have a frequency between one every other day and 8 per day 
	#        for more than half of the time when the disorder is active' as well

	# Initialize blank nodes for our restrictions
	symptom_bnode = BNode()
	location_bnode = BNode()
	duration_bnode = BNode()
	severity_bnode = BNode()
	intersection_node = BNode()

	# Modelled as an intersection of different Restriction classes
	g.add( (cluster, RDFS.subClassOf, intersection_node) )
	g.add( (intersection_node, OWL.intersectionOf, symptom_bnode))
	g.add( (intersection_node, OWL.intersectionOf, severity_bnode))
	g.add( (intersection_node, OWL.intersectionOf, location_bnode))
	g.add( (intersection_node, OWL.intersectionOf, duration_bnode))

	g.add( (symptom_bnode, RDF.type, OWL.Restriction) )
	g.add( (symptom_bnode, OWL.onProperty, concepts.symptom_predicate) )
	g.add( (symptom_bnode, OWL.someValuesFrom, symptom_list_object) )

	g.add( (severity_bnode, RDF.type, OWL.Restriction) )
	g.add( (severity_bnode, OWL.onProperty, concepts.intensity_predicate) )
	g.add( (severity_bnode, OWL.oneValueFrom, severity_list_object) )

	g.add( (location_bnode, RDF.type, OWL.Restriction) )
	g.add( (location_bnode, OWL.onProperty, concepts.location_predicate) )
	g.add( (location_bnode, OWL.oneValueFrom, location_list_object) )

	g.add( (duration_bnode, RDF.type, OWL.Restriction) )
	g.add( (duration_bnode, OWL.onProperty, concepts.duration_predicate) )
	g.add( (duration_bnode, OWL.oneValueFrom, duration_list_object) )

	# ####################################################### #
	#					MIGRAINE headache                     #
	# #########################################################
	migraine = URIRef(concepts.BASE_URL+'Migraine')
	g.add( (migraine, RDF.type, OWL.Class ) )

	# Link to SNOMED (from 'Headache disorder > Vascular headache')
	g.add( (migraine, OWL.sameAs, URIRef(PURL_PREFIX + '37796009')) )

	# Has one or more of the following symptoms
	blank_node = symptom_list_object = BNode()
	symptom_list = ['nausea', 'vomiting', 'photophobia', 'phonophobia',
					'aggravated']
	for i, symptom in enumerate(symptom_list):
		g.add( (blank_node, RDF.first, mappings.symptom_to_URI[symptom]) )
		blank_node2 = BNode()
		if i < len(symptom_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Moderate or severe pain intensity
	blank_node = severity_list_object = BNode()
	severity_list = ['moderate', 'severe']
	for i, severity in enumerate(severity_list):
		g.add( (blank_node, RDF.first, mappings.severity_to_URI[severity]) )
		blank_node2 = BNode()
		if i < len(severity_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Location is unilateral
	blank_node = location_list_object = BNode()
	location_list = ['unilateral']
	for i, location in enumerate(location_list):
		g.add( (blank_node, RDF.first, mappings.location_to_URI[location]) )
		blank_node2 = BNode()
		if i < len(location_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Headache attacks lasting 4-72 hr
	blank_node = duration_list_object = BNode()
	duration_list = ['H']
	for i, duration in enumerate(duration_list):
		g.add( (blank_node, RDF.first, mappings.duration_to_URI[duration]) )
		blank_node2 = BNode()
		if i < len(duration_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Characterisation of pain is pulsating
	blank_node = characterisation_list_object = BNode()
	characterisation_list = ['pulsating']
	for i, characterisation in enumerate(characterisation_list):
		g.add( (blank_node, RDF.first, mappings.characterisation_to_URI[characterisation]) )
		blank_node2 = BNode()
		if i < len(location_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Initialize blank nodes for our restrictions
	symptom_bnode = BNode()
	location_bnode = BNode()
	duration_bnode = BNode()
	severity_bnode = BNode()
	characterisation_bnode = BNode()
	intersection_node = BNode()

	# Modelled as an intersection of different Restriction classes
	g.add( (migraine, RDFS.subClassOf, intersection_node) )
	g.add( (intersection_node, OWL.intersectionOf, symptom_bnode))
	g.add( (intersection_node, OWL.intersectionOf, severity_bnode))
	g.add( (intersection_node, OWL.intersectionOf, location_bnode))
	g.add( (intersection_node, OWL.intersectionOf, duration_bnode))
	g.add( (intersection_node, OWL.intersectionOf, characterisation_bnode))

	g.add( (symptom_bnode, RDF.type, OWL.Restriction) )
	g.add( (symptom_bnode, OWL.onProperty, concepts.symptom_predicate) )
	g.add( (symptom_bnode, OWL.someValuesFrom, symptom_list_object) )

	g.add( (severity_bnode, RDF.type, OWL.Restriction) )
	g.add( (severity_bnode, OWL.onProperty, concepts.intensity_predicate) )
	g.add( (severity_bnode, OWL.oneValueFrom, severity_list_object) )

	g.add( (location_bnode, RDF.type, OWL.Restriction) )
	g.add( (location_bnode, OWL.onProperty, concepts.location_predicate) )
	g.add( (location_bnode, OWL.oneValueFrom, location_list_object) )

	g.add( (duration_bnode, RDF.type, OWL.Restriction) )
	g.add( (duration_bnode, OWL.onProperty, concepts.duration_predicate) )
	g.add( (duration_bnode, OWL.oneValueFrom, duration_list_object) )

	g.add( (characterisation_bnode, RDF.type, OWL.Restriction) )
	g.add( (characterisation_bnode, OWL.onProperty, concepts.characterisation_predicate) )
	g.add( (characterisation_bnode, OWL.oneValueFrom, characterisation_list_object) )

	# ####################################################### #
	#					TENSION headache                      #
	# #########################################################
	tension = URIRef(concepts.BASE_URL+'Tension')
	g.add( (tension, RDF.type, OWL.Class ) )

	# Link to SNOMED (from 'Headche disorder')
	g.add( (tension, OWL.sameAs, URIRef(PURL_PREFIX + '398057008')) )

	# Has NO MORE THAN ONE of the following symptoms
	blank_node = symptom_list_object = BNode()
	symptom_list = ['photophobia', 'phonophobia']  #  
	for i, symptom in enumerate(symptom_list):
		g.add( (blank_node, RDF.first, mappings.symptom_to_URI[symptom]) )
		blank_node2 = BNode()
		if i < len(symptom_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# TODO: Has no symptoms: ['vomiting', 'aggravated']

	# Mild or moderate intensity
	blank_node = severity_list_object = BNode()
	severity_list = ['mild', 'moderate']
	for i, severity in enumerate(severity_list):
		g.add( (blank_node, RDF.first, mappings.severity_to_URI[severity]) )
		blank_node2 = BNode()
		if i < len(severity_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Location is bilateral
	blank_node = location_list_object = BNode()
	location_list = ['bilateral']
	for i, location in enumerate(location_list):
		g.add( (blank_node, RDF.first, mappings.location_to_URI[location]) )
		blank_node2 = BNode()
		if i < len(location_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Headache attacks lasting hours to days
	blank_node = duration_list_object = BNode()
	duration_list = ['E', 'F', 'G', 'H', 'I', 'J']
	for i, duration in enumerate(duration_list):
		g.add( (blank_node, RDF.first, mappings.duration_to_URI[duration]) )
		blank_node2 = BNode()
		if i < len(duration_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Characterisation of pain is pressing (or tightening)
	blank_node = characterisation_list_object = BNode()
	characterisation_list = ['pressing']
	for i, characterisation in enumerate(characterisation_list):
		g.add( (blank_node, RDF.first, mappings.characterisation_to_URI[characterisation]) )
		blank_node2 = BNode()
		if i < len(location_list) - 1:
			g.add( (blank_node, RDF.rest, blank_node2) )
			blank_node = blank_node2

	# Initialize blank nodes for our restrictions
	symptom_bnode = BNode()
	location_bnode = BNode()
	duration_bnode = BNode()
	severity_bnode = BNode()
	characterisation_bnode = BNode()
	intersection_node = BNode()

	# Modelled as an intersection of different Restriction classes
	g.add( (tension, RDFS.subClassOf, intersection_node) )
	g.add( (intersection_node, OWL.intersectionOf, symptom_bnode))
	g.add( (intersection_node, OWL.intersectionOf, severity_bnode))
	g.add( (intersection_node, OWL.intersectionOf, location_bnode))
	g.add( (intersection_node, OWL.intersectionOf, duration_bnode))
	g.add( (intersection_node, OWL.intersectionOf, characterisation_bnode))

	g.add( (symptom_bnode, RDF.type, OWL.Restriction) )
	g.add( (symptom_bnode, OWL.onProperty, concepts.symptom_predicate) )
	g.add( (symptom_bnode, OWL.oneValueFrom, symptom_list_object) )

	g.add( (severity_bnode, RDF.type, OWL.Restriction) )
	g.add( (severity_bnode, OWL.onProperty, concepts.intensity_predicate) )
	g.add( (severity_bnode, OWL.oneValueFrom, severity_list_object) )

	g.add( (location_bnode, RDF.type, OWL.Restriction) )
	g.add( (location_bnode, OWL.onProperty, concepts.location_predicate) )
	g.add( (location_bnode, OWL.oneValueFrom, location_list_object) )

	g.add( (duration_bnode, RDF.type, OWL.Restriction) )
	g.add( (duration_bnode, OWL.onProperty, concepts.duration_predicate) )
	g.add( (duration_bnode, OWL.oneValueFrom, duration_list_object) )

	g.add( (characterisation_bnode, RDF.type, OWL.Restriction) )
	g.add( (characterisation_bnode, OWL.onProperty, concepts.characterisation_predicate) )
	g.add( (characterisation_bnode, OWL.oneValueFrom, characterisation_list_object) )

	g.serialize(destination='data/ICHD_KB.ttl', format='turtle')
	
	#return g
