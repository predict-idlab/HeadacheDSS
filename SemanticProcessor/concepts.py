from rdflib import URIRef

BASE_URL = 'http://chronicals.ugent.be/'

# Types/Classes
headache_type = URIRef(BASE_URL+'Headache')
symptom_type = URIRef(BASE_URL+'Symptom')
diagnose_type = URIRef(BASE_URL+'Diagnose')
characterisation_type = URIRef(BASE_URL+'Characterisation')
location_type = URIRef(BASE_URL+'Location')
severity_type = URIRef(BASE_URL+'Severity')
duration_group_type = URIRef(BASE_URL+'DurationGroup')

# Predicates
ub_predicate = URIRef(BASE_URL+'hasUpperBound')
lb_predicate = URIRef(BASE_URL+'hasLowerBound')
diagnose_predicate = URIRef(BASE_URL+'isType')
prev_attacks_predicate = URIRef(BASE_URL+'previousAttacks')
characterisation_predicate = URIRef(BASE_URL+'isCharacterizedByPain')
intensity_predicate = URIRef(BASE_URL+'hasIntensity')
location_predicate = URIRef(BASE_URL+'isLocated')
duration_predicate = URIRef(BASE_URL+'hasDuration')  # TODO: --> https://www.w3.org/TR/owl-time/ gebruiken
symptom_predicate = URIRef(BASE_URL+'hasSymptom')