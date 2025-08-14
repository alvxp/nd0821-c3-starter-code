# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
import uvicorn
import pandas as pd
import pickle
import os

app = FastAPI()

file_path = os.path.dirname(__file__)

model_path = os.path.join(file_path, "./model/trained_model.pkl")
model_used = pickle.load(open(model_path, "rb"))

encoder_path = os.path.join(file_path, "./model/encoder.pkl")
encoder = pickle.load(open(encoder_path, "rb"))

lb_path = os.path.join(file_path, "./model/lb.pkl")
lb = pickle.load(open(lb_path, "rb"))


class InputData(BaseModel):
    age: int = Field(None, examples=[39, 50])
    workclass: str = Field(None, examples=["State-gov", "Self-emp-not-inc"])
    fnlwgt: int = Field(None, examples=[77516, 83311])
    education: str = Field(None, examples=["Bachelors", "HS-grad"])
    education_num: int = Field(None, examples=[13, 9])
    marital_status: str = Field(None, examples=["Never-married", "Married-civ-spouse"])
    occupation: str = Field(None, examples=["Adm-clerical", "Exec-managerial"])
    relationship: str = Field(None, examples=["Not-in-family", "Husband"])
    race: str = Field(None, examples=["White", "Black"])
    sex: str = Field(None, examples=["Male", "Female"])
    capital_gain: int = Field(None, examples=[2174, 0])
    capital_loss: int = Field(None, examples=[0])
    hours_per_week: int = Field(None, examples=[40, 13])
    native_country: str = Field(None, examples=["United-States", "Mexico"])

@app.get("/")
async def say_hello():
    return {"message": "Hello World!"}

@app.post('/predict')
async def predict(data: InputData):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    sample = pd.DataFrame(data, index=[0])
    input_data = pd.DataFrame.from_dict(sample)

    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model=model_used, X=X)
    if prediction[0] > 0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K',
    data['prediction'] = prediction

    return data

if __name__ == '__main__':
    pass
