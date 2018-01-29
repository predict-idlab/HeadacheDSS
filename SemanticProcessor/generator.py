from rdflib import URIRef, BNode, Literal, Graph, Namespace
from rdflib.namespace import RDF, FOAF, OWL, RDFS, NamespaceManager
from rdflib.extras.infixowl import Restriction, Individual
import SemanticProcessor.concepts as concepts
import numpy as np


def generate_samples(class_name, graphs, n=10, output_path='../data/generated_samples.ttl', id_offset=10000):
	# Initialize the graph by parsing all given KG's
	g = Graph()
	for graph in graphs:
		g.parse(graph, format='turtle')

	# Double check whether class_name is a Diagnose and owl:Class
	qres = g.query("""SELECT ?diagnose WHERE {
						?diagnose rdf:type owl:Class .
						?diagnose rdf:type chron:Diagnose .
				   }
				   """, 
		 		   initNs={'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
		 		   		   'chron': Namespace('http://chronicals.ugent.be/'),
		 		   		   'owl': Namespace('http://www.w3.org/2002/07/owl#')},
		 		   initBindings={'diagnose': URIRef(concepts.BASE_URL + class_name)})
	
	if not len(list(qres)):  # Return None if a wrong class_name has been passed along
		return None

	# Filter out all the unique properties (except for rdf:type and chron:isType)
	qres = g.query("""SELECT distinct ?property WHERE{
						?headache rdf:type chron:Headache .
						?headache ?property ?obj .
						filter( ?property not in ( rdf:type, chron:isType ) ) .
					  }
				   """, initNs={'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
		 		   		   	    'chron': Namespace('http://chronicals.ugent.be/')})

	unique_properties = set()
	for row in qres:
		unique_properties.add(row[0])
	
	random_parameters_per_property = {}

	# First, lets generate values for oneValueFrom restrictions (np.random.choice of size=1)
	qres = g.query("""SELECT ?property ?item WHERE {
						?diagnose rdfs:subClassOf ?bnode1 .
						?bnode1 owl:intersectionOf ?bnode2 .
						?bnode2 rdf:type owl:Restriction .
						?bnode2 owl:onProperty ?property .
						?bnode2 owl:oneValueFrom/rdf:rest*/rdf:first ?item .
				   }
				   """,
				   initNs={'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
				   		   'rdfs': Namespace('http://www.w3.org/2000/01/rdf-schema#'),
		 		   		   'owl': Namespace('http://www.w3.org/2002/07/owl#')},
		 		   initBindings={'diagnose': URIRef(concepts.BASE_URL + class_name)})

	one_values_per_property = {}

	for row in qres:
		_property = row[0]
		if _property not in one_values_per_property:
			one_values_per_property[_property] = [row[1]]
		else:
			one_values_per_property[_property].append(row[1])

		unique_properties = unique_properties - set([_property])

	for prop in one_values_per_property:
		random_parameters_per_property[prop] = (one_values_per_property[prop], (1, 2), [1./len(one_values_per_property[prop])]*len(one_values_per_property[prop]))

	qres = g.query("""SELECT ?property ?item WHERE {
						?diagnose rdfs:subClassOf ?bnode1 .
						?bnode1 owl:intersectionOf ?bnode2 .
						?bnode2 rdf:type owl:Restriction .
						?bnode2 owl:onProperty ?property .
						?bnode2 owl:someValuesFrom/rdf:rest*/rdf:first ?item .
				   }
				   """,
				   initNs={'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
				   		   'rdfs': Namespace('http://www.w3.org/2000/01/rdf-schema#'),
		 		   		   'owl': Namespace('http://www.w3.org/2002/07/owl#')},
		 		   initBindings={'diagnose': URIRef(concepts.BASE_URL + class_name)})

	some_values_per_property = {}

	for row in qres:
		_property = row[0]
		if _property not in some_values_per_property:
			some_values_per_property[_property] = [row[1]]
		else:
			some_values_per_property[_property].append(row[1])

		unique_properties = unique_properties - set([_property])

	for prop in some_values_per_property:
		random_parameters_per_property[prop] = (some_values_per_property[prop], (1, len(some_values_per_property[prop]) + 1), 
											    [1./len(some_values_per_property[prop])]*len(some_values_per_property[prop]))

	for prop in unique_properties:
		qres = g.query("""SELECT ?value (count(?value) as ?count) WHERE {
							?headache chron:isType ?diagnose .
							?headache ?property ?value .
					   } GROUP BY ?value
					   """, initBindings={'diagnose': URIRef(concepts.BASE_URL + class_name),
					   					  'property': prop})

		value_counter = []
		for row in qres:
			value_counter.append((row[0], row[1].toPython()))

		total = sum([x[1] for x in value_counter])
		random_parameters_per_property[prop] = ([x[0] for x in value_counter], (1, 2), [x[1]/total for x in value_counter])

	new_graph = Graph()
	new_graph.bind('chron', Namespace(concepts.BASE_URL), override=True)

	for i in range(n):
		headache = URIRef(concepts.BASE_URL+'headache#'+str(id_offset+i))
		new_graph.add( (headache, RDF.type, concepts.headache_type) )
		new_graph.add( (headache, URIRef(concepts.BASE_URL+'isType'), URIRef(concepts.BASE_URL+class_name)) )

		for prop in random_parameters_per_property.keys():
			collection, size, p = random_parameters_per_property[prop]
			random_values = np.random.choice(collection, size=np.random.randint(size[0], size[1]), p=p)
			for value in random_values:
				if 'http' in value:
					new_graph.add((headache, prop, URIRef(value)))
				else:
					new_graph.add((headache, prop, Literal(value)))


	new_graph.serialize(destination=output_path, format='turtle')