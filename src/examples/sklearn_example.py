import os
import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flop_tracker import FlopTracker


def main():
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )

    clf = LogisticRegression(max_iter=1000)

    # Tracciamo FLOP del fit
    ft_fit = FlopTracker(run_name="sklearn_logreg_fit").sklearn_bind(
        model=clf,
        X=X,
        y=y,
        mode="fit",
        log_per_call=True,
        export_path="sklearn_logreg_fit_flop.csv",
        use_wandb=False,
    )

    # Tracciamo FLOP del predict
    ft_pred = FlopTracker(run_name="sklearn_logreg_predict").sklearn_bind(
        model=clf,
        X=X,
        mode="predict",
        log_per_call=True,
        export_path="sklearn_logreg_predict_flop.csv",
        use_wandb=False,
    )

    print("Fit FLOP:", ft_fit.total_flop)
    print("Predict FLOP:", ft_pred.total_flop)


if __name__ == "__main__":
    main()
