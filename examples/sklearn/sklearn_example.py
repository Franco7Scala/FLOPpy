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
# FLOPpy INTEGRATION
# ==========================================

# 1. Initialize the tracker with a custom experiment name
tracker = FLOPpyTracker(run_name="sklearn_test")

# 2. Start monitoring.
# For Scikit-learn, you only need to pass the model. FLOPpy will automatically
# wrap the underlying API to intercept inputs and calculate algorithmic workload.
tracker.run(model=model)


# ==========================================
# TRAINING AND INFERENCE
# ==========================================

# Train the model. FLOPpy dynamically intercepts the array shapes (200, 20)
# to calculate the FLOPs/BOPs required to build the 100 decision trees.
model.fit(X_train, y_train)

# Generate predictions. FLOPpy calculates the inference cost based on the
# test set dimensions (100, 20) and the average depth of the fitted forest.
preds = model.predict(X_test)


# ==========================================
# FINAL REPORT
# ==========================================

# 3. Generate and print the detailed hardware and algorithmic workload summary
report = tracker.report()
print(report)
