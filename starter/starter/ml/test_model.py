import os, sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score, fbeta_score
from starter.starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    preds = model.predict(X_train)

    # to check types and whether number of preds is equal to y_train
    assert isinstance(model, BaseEstimator)
    assert isinstance(model, ClassifierMixin)
    assert len(preds) == len(y_train)


def test_compute_model_metrics():
    y_true, y_pred = np.array([0, 1, 1, 0, 1, 0]), np.array([0, 1, 0, 0, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)

    assert np.isclose(precision, expected_precision)
    assert np.isclose(recall, expected_recall)
    assert np.isclose(fbeta, expected_fbeta)


def test_inference():
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    X_test = np.array([[1, 0], [0, 1]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    assert preds.shape[0] == X_test.shape[0]
