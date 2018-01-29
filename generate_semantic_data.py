from SemanticProcessor.encoder import encode
from SemanticProcessor.decoder import decode
from SemanticProcessor.kb import generate_ICHD_KB
import numpy as np
from rdflib import Graph
import pandas as pd

# Translate the original migbase dataset to a knowledge graph
# and generate OWL file with domain knowledge

# Encode KG
encode('data/migbase.csv', output_path='data/headache_KG.ttl')

# Generate the KB from the ICHD document
generate_ICHD_KB()