📖 Usage
========

Integration is transparent and does not require modifications to the model implementation.

PyTorch / Hugging Face Example
------------------------------

.. code-block:: python

   import torch.nn as nn
   from floppy import FLOPpyTracker
   from transformers import AutoModel

   wandb_config = WandbConfiguration(
     project_name="your_experiment",
     group_name="your_group",
     reporter_key="your_wandb_key_here"
   )

   # 1. Define your model, loss and optimizer
   model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters())
   num_epochs = 10

   # 2. Initialize the tracker
   tracker = FLOPpyTracker(run_name="pytorch_experiment")

   # 3. Run monitoring
   tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)

   # 4. Do something with the model
   for _ in range(num_epochs):
       for xb, yb in your_data_loader:
           optimizer.zero_grad()
           y_hat = model(xb)
           loss = loss_fn(y_hat, yb)
           loss.backward()
           optimizer.step()
           tracker.batch()

       tracker.epoch()

   # 5. Access the report
   report = tracker.report()
   print(report)

Scikit-learn Example
--------------------

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from floppy import FLOPpyTracker

   # 1. Define your model
   model = RandomForestClassifier(n_estimators=100)

   # 2. Initialize the tracker
   tracker = FLOPpyTracker(run_name="sklearn_test")

   # 3. Run monitoring
   tracker.run(model=model)

   # 4. Do something with the model
   model.fit(X_train, y_train)
   preds = model.predict(X_test)

   # 5. Access the report
   report = tracker.report(print_summary=True)