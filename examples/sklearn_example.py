import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from floppy import FLOPpyTracker

X, y = make_classification(
    n_samples=300,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    random_state=42,
)

X_train, y_train = X[:200], y[:200]
X_test, y_test = X[200:], y[200:]

# ============================================================
# MODE 1
# ============================================================

model = RandomForestClassifier(n_estimators=100)

tracker = FLOPpyTracker(run_name="sklearn_test")
tracker.run(model=model)

model.fit(X_train, y_train)
preds = model.predict(X_test)

report = tracker.report()

# ============================================================
# MODE 2: context manager style
# ============================================================

model = RandomForestClassifier(n_estimators=100)

with FLOPpyTracker(run_name="sklearn_test_with") as tracker:
    tracker.start(model=model)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    tracker.stop()

report = tracker.report()
