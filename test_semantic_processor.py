from SemanticProcessor.encoder import encode
from SemanticProcessor.decoder import decode
from SemanticProcessor.kb import generate_ICHD_KB
import numpy as np
from rdflib import Graph
import pandas as pd

# Test whether decode(encode(data)) == data

# Encode KG
encode('data/migbase.csv', output_path='data/headache_KG.ttl')

# Decode the KG
g = Graph()
g.parse("data/headache_KG.ttl", format="turtle")
df = decode(g)

# Read original file
migbase = pd.read_csv('data/migbase.csv').reset_index()
_columns = [col for col in migbase.columns
            if len(np.unique(migbase[col])) > 1]
migbase = migbase[_columns]
migbase = migbase.drop([
	'headache_with_aura', 'pericranial', 'vertigo', 'aura_duration', 
	'diplopia', 'ataxia', 'agitation', 'headache_days', 'aura_development'
], axis=1)

# Sort by index such that rows are in same order
df = df.sort_values(by='index')
migbase = migbase.sort_values(by='index')

# Reindex the columns so they are in same order
df = df[migbase.columns]  

# Test equality of original and decode(encode(original))
np.testing.assert_array_equal(migbase.values, df.values)