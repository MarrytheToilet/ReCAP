import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('grade',Path(__file__).resolve().parents[1]/'scripts/tau2_grade_validation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_empty_or_missing_results_cannot_be_success():
    for text in ['{}','{"results": []}','{"met": true}']:
        with pytest.raises(ValueError):m.validate_judge_output(text,['required'])

def test_boolean_and_identity_checked():
    for text in ['{"results":[{"expectedOutcome":"required","metExpectation":"false","reasoning":"x"}]}','{"results":[{"expectedOutcome":"other","metExpectation":true,"reasoning":"x"}]}']:
        with pytest.raises(ValueError):m.validate_judge_output(text,['required'])

def test_valid_negative_is_preserved():
    import json
    result=m.validate_judge_output('{"results":[{"expectedOutcome":"required","metExpectation":false,"reasoning":"not met"}]}',['required'])
    assert json.loads(result)['results'][0]['metExpectation'] is False
