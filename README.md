# FLOPpy: A hardware-agnostic Python library to monitor the computational cost of Machine and Deep Learning algorithms 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**FLOPpy** is a Python library designed to estimate and monitor the computational workload of Machine Learning (ML) and Deep Learning (DL) models. By systematically counting **Floating Point Operations (FLOPs)** and **BOPs (Bit-OPerations)**, the library enables a hardware-independent assessment of the algorithmic workload. Unlike metrics such as execution time or energy consumption, FLOPs reflect the intrinsic computational demand of an algorithm, ensuring comparability and reproducibility across different systems.

---

## 🚀 Key Features

* **Hardware-Agnostic Monitoring**: Provides a standardized measure of computational demand that does not depend on specific hardware characteristics or infrastructure;
* **Broad Framework Support**: Offers a unified interface for models implemented in **PyTorch** (including **Hugging Face** models) and **Scikit-learn**;
* **Comprehensive Pipeline Tracking**: Beyond simple forward passes, it accounts for **backward passes**, **optimizer updates**, **loss evaluations**, **tokenizer operations** and so on;
* **Modular Architecture**: Designed with a provider pattern and structural decoupling, allowing easy extension to other backends;
* **Real-time Integration**: Supports seamless synchronization with **Weights & Biases (WandB)** for real-time visualization.

---

## 📦 Installation

```bash 
pip install floppy-lib
```

## Dependencies

The library requires the following environment and tools:

* **Python**: Core language.
* **NumPy**: Used for multidimensional array manipulation and analytical complexity formulas.
* **Scikit-Learn**: Supported for monitoring classical machine learning algorithms.
* **PyTorch**: Supported for deep learning tracking via high-level hooks and low-level ATen dispatching.
* **psutil**: Essential for capturing detailed hardware snapshots, including CPU cores, RAM, and system usage.
* **Wandb**: Used for real-time visualization and remote experiment tracking.

---

## 📖 Usage

Integration is transparent and does not require modifications to the model implementation.

### PyTorch / Hugging Face Example

```python
import torch.nn as nn
from floppy import FLOPpyTracker, WandbConfiguration
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
```

### Scikit-learn Example

```python
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
```

## 🔬 Methodology

FLOPpy employs distinct strategies to ensure computational accuracy:

* **PyTorch (Hybrid Methodology):** Synergistically combines two approaches to capture the full training lifecycle:
  * **Module Hooks:** Uses `TorchTrainingHooks` to register forward hooks on standard neural network layers, dynamically intercepting tensor shapes to apply theoretically derived formulas for components like convolutions and linear projections;
  * **Low-level Dispatching:** Deploys `TorchDispatchMode` to intercept underlying C++ ATen dispatch calls. This captures "loose" operations outside of `nn.Module` objects, such as residual skip connections, loss computations, and in-place tensor manipulations executed by optimizers during the backward pass;
* **Scikit-learn:** Implements a dynamic method-wrapping strategy through the `SklearnBackend`. It intercepts standard API methods (e.g., `fit()`, `predict()`) to extract input/output array dimensions and applies targeted algorithmic complexity formulas.

---

## 📊 Output Metrics

The `FLOPpyReport` object provides detailed statistics:

* **`model_flops`**: The core algorithmic cost derived from the model's structural layers;
* **`optimizer_flops`**: Computational overhead introduced by the optimization step, such as gradient updates and momentum calculations;
* **`loss_flops`**: Operations related to the loss function computation;
* **`model_bops`**: The precision-aware hardware computational effort of the model's architecture, calculated as Bit-Operations (BOPs) to reflect the underlying data types (e.g. INT8 vs. FP32);
* **`optimizer_bops`**: The hardware-level computational cost of the optimization step, accounting for the specific bit-width used during gradient updates and weight adjustments;
* **`loss_bops`**: The actual hardware effort required for the loss function evaluation, measured in Bit-Operations based on the tensor precision;
* **`preproc_ops`**: Workload from input preparation, such as tokenizer operations for Large Language Models;
* **`SystemInfo`**: A detailed snapshot of the execution environment, including CPU/GPU specifications, RAM, and OS version.

---

## ✍️ Authors & Citation

**Francesco Scala, Francesco Mandarino, Liliana Martirano, and Luigi Pontieri.** *Institute of High Performance Computing and Networking (ICAR-CNR) & University of Calabria, Italy.*

If you use FLOPpy in your research, please cite:

> Coming soon...

---

## 📄 License

This software is licensed under the **GNU Public License v3.0 (GPL3)**.