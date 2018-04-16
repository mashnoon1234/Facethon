import json 
from pprint import pprint

data = json.load(open('config.json'))

pprint(data)