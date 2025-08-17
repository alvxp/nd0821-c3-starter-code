# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference
import uvicorn
import pandas as pd
import pickle
import os

app = FastAPI()

file_path = os.path.dirname(__file__)
save_path = os.path.join(file_path, "model")

model_path = os.path.join(save_path, "trained_model.pkl")
model_used = pickle.load(open(model_path, "rb"))

encoder_path = os.path.join(save_path, "encoder.pkl")
encoder = pickle.load(open(encoder_path, "rb"))

lb_path = os.path.join(save_path, "lb.pkl")
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
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    sample = pd.DataFrame([data.dict()])

    X, _, _, _ = process_data(
        sample,
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
        prediction = '<=50K'

    return {**data.dict(), "prediction": prediction}


if __name__ == '__main__':
    pass
