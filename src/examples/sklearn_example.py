import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker
from trainers import train_sklearn


def main():
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=1000)

    # FIT
    FlopTracker(run_name="sklearn_logreg_fit", print_summary=True).run(
        model=model,
        backend="sklearn",
        train_fn=train_sklearn,
        train_kwargs={"model": model, "mode": "fit", "X": X, "y": y},
        log_per_batch=True,
        export_path="sklearn_logreg_fit_flop.csv",
        use_wandb=False,
    )

    # PREDICT
    FlopTracker(run_name="sklearn_logreg_predict", print_summary=True).run(
        model=model,
        backend="sklearn",
        train_fn=train_sklearn,
        train_kwargs={"model": model, "mode": "predict", "X": X},
        log_per_batch=True,
        export_path="sklearn_logreg_predict_flop.csv",
        use_wandb=False,
    )


if __name__ == "__main__":
    main()
