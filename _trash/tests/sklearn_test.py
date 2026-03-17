import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from floppy.tracker import FLOPpyTracker


def test_logistic_regression():
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        random_state=0,
    )

    model = LogisticRegression(max_iter=100)

    tracker = FLOPpyTracker(
        run_name="sklearn_test",
        print_summary=True,
    )

    with tracker.run(
        model=model,
        export_path="sklearn_test.csv",
    ):
        model.fit(X, y)
        model.predict(X)

    rep = tracker.report

    assert rep.model_flop > 0
    assert rep.optimizer_flop == 0
    assert rep.loss_forward_flop == 0
    assert rep.loss_backward_flop == 0


def test_transform():
    X = np.random.randn(200, 10)
    scaler = StandardScaler()

    tracker = FLOPpyTracker(
        run_name="sklearn_transform",
        print_summary=True,
    )

    with tracker.run(
        model=scaler,
        export_path="sklearn_transform.csv",
    ):
        scaler.fit(X)
        scaler.transform(X)

    rep = tracker.report

    assert rep.model_flop >= 0


def main():
    test_logistic_regression()
    test_transform()


if __name__ == "__main__":
    main()
