import requests

url = "http://127.0.0.1:8000/predict"

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

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response JSON:", response.json())