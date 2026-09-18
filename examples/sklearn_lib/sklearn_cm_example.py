from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from floppy import FLOPpyTracker


# ==========================================
# DATA SETUP
# ==========================================

# Generate a synthetic tabular dataset for classification
X, y = make_classification(
    n_samples=300,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    random_state=42,
)

# Split the dataset into training (200 samples) and testing (100 samples)
X_train, y_train = X[:200], y[:200]
X_test, y_test = X[200:], y[200:]

# ==========================================
# MODEL SETUP
# ==========================================

# Define a standard Scikit-learn model
# FLOPpy supports tree-based models, linear models, SVMs, clustering, etc.
model = RandomForestClassifier(n_estimators=100)

# ==========================================
# FLOPpy INTEGRATION (CONTEXT MANAGER)
# ==========================================

# Using FLOPpyTracker as a context manager (with statement) ensures safe
# execution and clean scope management for your profiling session.
with FLOPpyTracker(run_name="sklearn_test_with") as tracker:
    # 1. Start monitoring by passing the model.
    # FLOPpy will automatically wrap the underlying API to intercept inputs.
    tracker.start(model=model)

    # ==========================================
    # TRAINING AND INFERENCE
    # ==========================================

    # Train the model. FLOPpy dynamically intercepts the array shapes (200, 20)
    # to calculate the FLOPs/BOPs required to build the 100 decision trees.
    model.fit(X_train, y_train)

    # Generate predictions. FLOPpy calculates the inference cost based on the
    # test set dimensions (100, 20) and the average depth of the fitted forest.
    preds = model.predict(X_test)

    # Explicitly stop the tracking before exiting the context manager scope.
    # This safely un-wraps the Scikit-learn model methods.
    tracker.stop()

# ==========================================
# FINAL REPORT
# ==========================================

# 2. Generate and print the detailed hardware and algorithmic workload summary.
# This is safely called outside the 'with' block.
report = tracker.report()
print(report)
