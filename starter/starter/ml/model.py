from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    hg_model = HistGradientBoostingClassifier(max_iter=1000, random_state=23)
    hg_model.fit(X_train, y_train)
    return hg_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds



# def output_slices(df, features, y, preds):
    """ Function for calculating descriptive stats on slices of the Iris dataset."""
    for var in df[features].unique():
        clean_df = df[df[features] == var]
        mean = df_temp[features].mean()
        stddev = df_temp[features].std()
        print(f"Class: {var}")
        print(f"{features} mean: {mean:.4f}")
        print(f"{features} stddev: {stddev:.4f}")
    print()

