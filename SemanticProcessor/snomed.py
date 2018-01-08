import json
from rdflib import URIRef, BNode, Literal, Graph, Namespace
from rdflib.namespace import RDF, FOAF, OWL, RDFS, NamespaceManager
from rdflib.extras.infixowl import Restriction, Individual
import pandas as pd
import numpy as np
from urllib.request import urlopen, quote
import SemanticProcessor.mappings as mappings
import SemanticProcessor.concepts as concepts

def getDescriptionsByString(searchTerm, semTag='finding'):
	baseUrl = 'http://127.0.0.1:3000/snomed/'
	edition = 'en-edition'
	version = 'v20170731'
	url = baseUrl + edition + '/' + version + '/descriptions?query=' + quote(searchTerm) + '&limit=50&searchMode=partialMatching&lang=english&statusFilter=activeOnly&skipTo=0&returnLimit=100&semanticFilter='+quote(semTag)+'&normalize=true'
	response = urlopen(url).read()
	data = json.loads(response.decode('utf-8'))
	#	print(searchTerm, data)
	if len(data['matches']):
	    return 'http://purl.bioontology.org/ontology/SNOMEDCT/' + data['matches'][0]['conceptId']  # Already sorted on levensteihn distance between term and searchTerm
	else:  # No matches found with the semTag, try without
	    url = baseUrl + edition + '/' + version + '/descriptions?query=' + quote(searchTerm) + '&limit=50&searchMode=partialMatching&lang=english&statusFilter=activeOnly&skipTo=0&returnLimit=100&normalize=true'
	    response = urlopen(url).read()
	    data = json.loads(response.decode('utf-8'))
	    return 'http://purl.bioontology.org/ontology/SNOMEDCT/' + data['matches'][0]['conceptId'] 