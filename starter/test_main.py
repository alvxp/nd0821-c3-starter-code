import os, sys
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))

from main import app

client = TestClient(app)


def test_get_path_query():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': "Hello World!"}


def test_post_pos():
    data = {"age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlwgt": 209642,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States"
            }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"

def test_post_neg():
    data = {"age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_post_malformed():
    data = {"age": "Twenty"}
    r = client.post("/predict", json=json.dumps(data))
    assert r.status_code != 200
