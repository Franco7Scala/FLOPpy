# FLOPpy: A hardware-agnostic Python library to monitor the computational cost of Machine and Deep Learning algorithms 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**FLOPpy** is a Python library designed to estimate and monitor the computational workload of Machine Learning (ML) and Deep Learning (DL) models. By systematically counting **Floating Point Operations (FLOPs)**, the library enables a hardware-independent assessment of the algorithmic workload. Unlike metrics such as execution time or energy consumption, FLOPs reflect the intrinsic computational demand of an algorithm, ensuring comparability and reproducibility across different systems.

---

## 🚀 Key Features

* **Hardware-Agnostic Monitoring**: Provides a standardized measure of computational demand that does not depend on specific hardware characteristics or infrastructure;
* **Broad Framework Support**: Offers a unified interface for models implemented in **PyTorch** (including also **Hugging Face** models) and **Scikit-learn**;
* **Comprehensive Pipeline Tracking**: Beyond simple forward passes, it accounts for **backward passes**, **optimizer updates**, **loss evaluations**, **tokenizer operations** and so on;
* **Modular Architecture**: Designed with a provider pattern and structural decoupling, allowing easy extension to other backends;
* **Real-time Integration**: Supports seamless synchronization with **Weights & Biases (WandB)** for real-time visualization.

---

## 📦 Installation

```bash
pip install floppy
```

## Dependencies

The library requires the following environment and tools:

* **Python** (Core language)
* **NumPy** (Array operations and complexity formulas)
* **Scikit-Learn** (Classical ML tracking)
* **PyTorch** (Deep learning tracking via ATen dispatching)
* **Wandb** (Optional real-time logging)

---

## 📖 Usage

Integration is transparent and does not require modifications to the model implementation.

### PyTorch / Hugging Face Example

```python
import torch.nn as nn
from floppy import FLOPpyTracker
from transformers import AutoModel

# 1. Define your model, loss and optimizer
model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 2. Initialize the tracker
tracker = FLOPpyTracker(
    run_name="pytorch_experiment",
    print_summary=True,
    print_hardware=True
)

# 3. Run monitoring
tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)

# 4. Do something with the model
y_hat = model(input_data)
loss = loss_fn(y_hat, labels)
loss.backward()
optimizer.step()

# 5. Access the report
print(tracker.report())
```

### Scikit-learn Example

```python
from sklearn.ensemble import RandomForestClassifier
from floppy import FLOPpyTracker

# 1. Define your model
model = RandomForestClassifier(n_estimators=100)

# 2. Initialize the tracker
tracker = FLOPpyTracker(
    run_name="sklearn_test",
    print_summary=True,
    print_hardware=True
)

# 3. Run monitoring
tracker.run(model=model)

# 4. Do something with the model
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 5. Access the report
print(tracker.report())
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
* **`preproc_ops`**: Workload from input preparation, such as tokenizer operations for Large Language Models;
* **`HardwareInfo`**: A detailed snapshot of the execution environment, including CPU/GPU specifications, RAM, and OS version.

---

## ✍️ Authors & Citation

**Francesco Scala, Francesco Mandarino, Liliana Martirano, and Luigi Pontieri.** *Institute of High Performance Computing and Networking (ICAR-CNR) & University of Calabria, Italy.*

If you use FLOPpy in your research, please cite:

> TODO after publication

---

## 📄 License

This software is licensed under the **GNU Public License v3.0 (GPL3)**.