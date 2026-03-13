import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from floppy_tracker import FLOPpyTracker


X, y = make_classification(
    n_samples=300,
    n_features=20,
    random_state=0
)

model = LogisticRegression(max_iter=100)

tracker = FLOPpyTracker(
    run_name="sklearn_example",
    print_summary=True
)

with tracker.run(
    model=model,
    export_path="sklearn_example.csv"
):

    model.fit(X, y)

    preds = model.predict(X)


X = np.random.randn(200, 10)

scaler = StandardScaler()

tracker = FLOPpyTracker(
    run_name="sklearn_transform_example",
    print_summary=True
)

with tracker.run(
    model=scaler,
):

    scaler.fit(X)

    X_scaled = scaler.transform(X)
