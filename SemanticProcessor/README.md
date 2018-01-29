# SemanticProcessor

This code contains all code in order to translate the migbase CSV file to a semantically annotated knowledge base and vice
versa (from the KB back to a CSV with features). It consists of the following files:

  * **concepts: contains hardcoded URI's, to be used as elements in triples**
  * **mappings: define dicts that maps the URI's to python objects (str, int, ...)**
  * **kb: contains `generate_ICHD_kb()` that generates a knowledge graph based on the ICHD document**
  * **generator: generate artificial samples of a certain class (used for oversampling)**
  * **encoder: translate migbase feature matrix to knowledge graph**
  * **decoder: translate knowledge graph to feature matrix**
  * **serve: can be used to browse a given rdf file in the browser**
