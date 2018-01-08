from encoder import encode
from decoder import decode
import numpy as np
from rdflib import Graph
import pandas as pd

# Encode KG
encode('data/migbase.csv', output_path='data/headache_KG.ttl')

# Decode the KG
g = Graph()
g.parse("data/headache_KG.ttl", format="turtle")
df = decode(g)

# Read original file
migbase = pd.read_csv('data/migbase.csv').reset_index()

# Sort by index such that rows are in same order
df = df.sort_values(by='index')
migbase = migbase.sort_values(by='index')

# Reindex the columns so they are in same order
df = df[migbase.columns]  

# Test equality of original and decode(encode(original))
np.testing.assert_array_equal(migbase.values, df.values)