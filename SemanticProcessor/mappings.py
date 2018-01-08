"""
Mappings in this file (dicts):
------------------------------

* symptoms
* wrongly_written_symptoms, wrongly_written_symptoms_revert
* symptom_to_URI, URI_to_symptom
* diagnoses_to_URI, URI_to_diagnoses
* characterisation_to_URI, URI_to_characterisation
* location_to_URI, URI_to_location
* severity_to_URI, URI_to_severity
* duration_to_URI, URI_to_duration

"""

from rdflib import URIRef

BASE_URL = 'http://chronicals.ugent.be/'

symptoms = ['nausea', 'vomiting', 'photophobia', 'phonophobia', 'aggravated', 'conjunctival_injection',
		  	'lacrimation', 'nasal_congestion', 'rhinorrhoea', 'eyelid_oedema', 'sweating', 'miosis',
		  	'ptosis', 'speech_disturbance', 'visual_symptoms', 'sensory_symptoms', 'homonymous',
		  	'dysarthria', 'hemiplegic']
wrongly_written_symptoms = {'vomiting': 'vomitting', 'homonymous': 'homonymous_symptomps',
 							'sensory_symptoms': 'sensory_symptomps', 'aggravated': 'aggravation',
 							'visual_symptoms': 'visual_symptomps'}
symptom_to_URI = {}
symptom_type = URIRef(BASE_URL+'Symptom')
for symptom in symptoms:
	uri_suffix = '_'.join([x[0].upper()+x[1:] for x in symptom.split('_')])
	symptom_to_URI[symptom] = URIRef(BASE_URL+uri_suffix)

wrongly_written_symptoms_revert = {}
for key in wrongly_written_symptoms:
	wrongly_written_symptoms_revert[wrongly_written_symptoms[key]] = key

URI_to_symptom = {}
for key in symptom_to_URI:
	URI_to_symptom[symptom_to_URI[key]] = key

diagnoses_to_URI = {
	'migraine': URIRef(BASE_URL+'Migraine'),
	'cluster': URIRef(BASE_URL+'Cluster'),
	'tension': URIRef(BASE_URL+'Tension')
}

URI_to_diagnoses = {}
for key in diagnoses_to_URI:
	URI_to_diagnoses[diagnoses_to_URI[key]] = key

characterisation_to_URI = {
	'pressing': URIRef(BASE_URL+'Pressing'),
	'pulsating': URIRef(BASE_URL+'Pulsating'),
	'stabbing': URIRef(BASE_URL+'Stabbing')
}

URI_to_characterisation = {}
for key in characterisation_to_URI:
	URI_to_characterisation[characterisation_to_URI[key]] = key

location_to_URI = {
	'unilateral': URIRef(BASE_URL+'Unilateral'),
	'bilateral': URIRef(BASE_URL+'Bilateral'),
	'orbital': URIRef(BASE_URL+'Orbital'),
}

URI_to_location = {}
for key in location_to_URI:
	URI_to_location[location_to_URI[key]] = key


severity_to_URI = {
	'very severe': URIRef(BASE_URL+'Very_Severe'),
	'severe': URIRef(BASE_URL+'Severe'),
	'moderate': URIRef(BASE_URL+'Moderate'),
	'mild': URIRef(BASE_URL+'Mild'),
}

URI_to_severity = {}
for key in severity_to_URI:
	URI_to_severity[severity_to_URI[key]] = key

duration_to_URI = {
	'A': URIRef(BASE_URL+'DurationGroupA'),
	'B': URIRef(BASE_URL+'DurationGroupB'),
	'C': URIRef(BASE_URL+'DurationGroupC'),
	'D': URIRef(BASE_URL+'DurationGroupD'),
	'E': URIRef(BASE_URL+'DurationGroupE'),
	'F': URIRef(BASE_URL+'DurationGroupF'),
	'G': URIRef(BASE_URL+'DurationGroupG'),
	'H': URIRef(BASE_URL+'DurationGroupH'),
	'I': URIRef(BASE_URL+'DurationGroupI'),
	'J': URIRef(BASE_URL+'DurationGroupJ'),
}
URI_to_duration = {}
for key in duration_to_URI:
	URI_to_duration[duration_to_URI[key]] = key