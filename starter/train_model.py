# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.
import os
import pandas as pd
import pickle


# Add code to load in the data.
file_path = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(file_path, "data", "census.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train and save a model.
new_model = train_model(X_train, y_train)
save_path = os.path.join(file_path, "model")

model_path = os.path.join(save_path, "trained_model.pkl")
pickle.dump(new_model, open(model_path, "wb"))

encoder_path = os.path.join(save_path, "encoder.pkl")
pickle.dump(encoder, open(encoder_path, "wb"))

lb_path = os.path.join(save_path, "lb.pkl")
pickle.dump(lb, open(lb_path, "wb"))

preds = inference(new_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
