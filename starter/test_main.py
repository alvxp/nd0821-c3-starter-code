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
    data = {"age": 31,
            "workclass": "Private",
            "fnlwgt": 45781,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 14084,
            "capital_loss": 0,
            "hours_per_week": 50,
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
