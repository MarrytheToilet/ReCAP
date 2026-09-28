"""Validate NL-judge outputs before upstream tau2 multiplies their rewards."""
import json
from collections import Counter

def validate_judge_output(text, expected):
    value=text.strip()
    if value.startswith('```'):
        value=value.split('\n',1)[1].rsplit('```',1)[0]
    data=json.loads(value)
    if not isinstance(data,dict) or not isinstance(data.get('results'),list):
        raise ValueError('Judge must return an object with results array')
    rows=data['results']
    if len(rows)!=len(expected):
        raise ValueError('Judge omitted or added assertions')
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('expectedOutcome'),str):
            raise ValueError('Missing expectedOutcome')
        if type(row.get('metExpectation')) is not bool:
            raise ValueError('metExpectation must be boolean')
        if not isinstance(row.get('reasoning'),str):
            raise ValueError('Missing reasoning')
    if Counter(r['expectedOutcome'] for r in rows)!=Counter(expected):
        raise ValueError('Judge changed assertion identities')
    return json.dumps(data)
