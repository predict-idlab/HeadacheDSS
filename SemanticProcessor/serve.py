import rdflib
from rdflib_web.lod import serve
	
g = rdflib.Graph()
g.parse("ICHD_KB.ttl", format="turtle")

serve(g) 