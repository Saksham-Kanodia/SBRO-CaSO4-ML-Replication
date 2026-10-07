import numpy as np

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import build_m52_gpr


def evaluate_m52_gpr(X, y, random_state=42):
    """
    Evaluate the M5/2 Gaussian Process Regression model
    using 5-fold cross-validation.

    NOTE:
    This is currently the experimental implementation.
    The exact paper replication configuration will be
    validated against the reported results.
    """

    model = build_m52_gpr()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", model)
    ])

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        n_jobs=1
    )

    r2 = r2_score(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "predictions": predictions
    }