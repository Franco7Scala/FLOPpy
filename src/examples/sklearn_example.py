import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker
from tracker import Tracker
from observers import BatchContext

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_sklearn_observed(*, model, X_train, y_train, X_test, observers=None):
    """
    Training esterno per sklearn che notifica manualmente gli observer.
    sklearn non ha forward hooks, quindi questo esempio valida:
    - architettura observer
    - logging / export / wandb (se attivi)
    - gestione "chiamate" come batch
    """
    observers = observers or []

    # start training
    for obs in observers:
        obs.on_train_start({"backend": "sklearn"})

    # epoca 0
    epoch = 0
    for obs in observers:
        obs.on_epoch_start(epoch)

    # "batch" = fit
    bc = BatchContext(epoch=epoch, batch_idx=0, batch_size=len(X_train))
    for obs in observers:
        obs.on_batch_start(bc)

    model.fit(X_train, y_train)

    for obs in observers:
        obs.on_batch_end(bc)

    # "batch" = predict
    bc2 = BatchContext(epoch=epoch, batch_idx=1, batch_size=len(X_test))
    for obs in observers:
        obs.on_batch_start(bc2)

    _ = model.predict(X_test)

    for obs in observers:
        obs.on_batch_end(bc2)

    for obs in observers:
        obs.on_epoch_end(epoch)

    for obs in observers:
        obs.on_train_end({"backend": "sklearn"})


def main():
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)

    # Facade: crea il Tracker e lo passa a train_fn come observer
    ft = FlopTracker(run_name="sklearn_facade_observer", print_summary=True).run(
        model=clf,
        backend="sklearn",  
        train_fn=train_sklearn_observed,
        train_kwargs=dict(
            model=clf,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        ),
        log_per_batch=True,
        log_per_epoch=True,
        export_path="sklearn_facade_observer.csv",
        use_wandb=False,
    )

    print("REPORT:", ft.report)


if __name__ == "__main__":
    main()
