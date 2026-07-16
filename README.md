# FLOPpy: A hardware-agnostic Python library to monitor the computational cost of Machine and Deep Learning algorithms 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.1-orange.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/floppy-tracker?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/floppy-tracker)

**FLOPpy** is a versatile Python library designed to monitor and estimate the algorithmic workload of both **Deep Learning (PyTorch)** and **Machine Learning (Scikit-learn)** models. 

By systematically tracking **Floating Point Operations (FLOPs)** and **BOPs (Bit-OPerations)**, it provides a hardware-independent assessment of the total computational demand, spanning from standard **Forward** and **Backward** passes to **Optimizer updates** and **Loss** evaluations.

## 🚀 Key Features

* **Hardware-Agnostic Monitoring**: Provides a standardized measure of computational demand that does not depend on specific hardware characteristics or infrastructure;
* **Cross-Framework Support**: Seamlessly profile models from `torch` (including `Hugging Face` models) and `scikit-learn` using a unified API;
* **Modular Architecture**: Designed with a provider pattern and structural decoupling, allowing easy extension to other backends; 
* **Full Pipeline Tracking**: Go beyond simple inference, monitor the cost of training (Backward pass), Loss computation, Optimizer steps, and even pre-processing operations like tokenization;
* **Transparent Integration**: Zero-boilerplate integration via a non-intrusive, hook-based architecture and safe monkey-patching;
* **The "Escape Hatch"**: Native support for tracking **quantized layers** (e.g., 4-bit, 8-bit) and **fused/custom optimizers** (BitsAndBytes, Apex, DeepSpeed) that typically bypass standard profilers;
* **Reproducibility**: Unlike execution time or energy metrics, FLOPs and BOPs reflect the intrinsic complexity of an algorithm, ensuring consistent results across different systems;
* **Real-time Integration**: Supports seamless synchronization with **Weights & Biases (WandB)** for real-time visualization.

## 📊 Why FLOPpy?

In an era of large-scale models and specialized hardware, execution time is no longer a sufficient metric for efficiency. **FLOPpy** allows researchers and developers to:
1. Compare the efficiency of different architectures regardless of the GPU/CPU used;
2. Quantify the real computational savings of quantization (FP16 vs INT8 vs INT4);
3. Identify bottlenecks in the training loop, including the often-overlooked optimizer overhead.

---

## 📦 Installation

```bash 
pip install floppy-tracker
```

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

### 🛠️ Computational Strategy & Backends

**FLOPpy** employs high-precision, transparent strategies across different frameworks to ensure maximum accuracy without requiring any changes to the user's original code.

#### **PyTorch: Unified Dispatch & Patching**
The library avoids the overhead and limitations of traditional per-module hooks by operating directly at the functional and tensor level:

* **Root Hooks & Low-Level Dispatching:** Instead of attaching hooks to every single sub-module, FLOPpy attaches a single boundary hook to the root model. Inside this forward pass, it deploys `TorchDispatchMode` via the `UniversalFlopCounter` to intercept underlying C++ ATen dispatch calls in real-time. This captures all mathematical operations, including those occurring outside of standard `nn.Module` objects, such as residual skip connections and element-wise tensor manipulations;
* **Transparent Backward Tracking:** Implements safe monkey-patching of `torch.Tensor.backward`. This encapsulates the entire Autograd graph execution within a tracking context, overcoming the well-known architectural limitations of standard PyTorch backward hooks on container modules (e.g., `nn.Sequential`);
* **Optimizer & Loss Hooks:** Utilizes targeted `TorchTrainingHooks` to intercept `optimizer.step()` calls and loss function evaluations. It features a specialized **"Escape Hatch"** fallback logic to accurately estimate the workload of fused or quantized optimizers (e.g., *BitsAndBytes*, *Apex*, *DeepSpeed*) that execute custom C++/CUDA kernels and bypass the standard PyTorch dispatcher.

#### **Scikit-Learn: Dynamic API Wrapping**
The `SklearnBackend` implements a non-intrusive method-wrapping strategy to seamlessly support classical Machine Learning workflows:

* **Method Interception:** Automatically wraps standard API methods—`fit()`, `predict()`, and `transform()`—to extract input and output array dimensions at runtime.
* **Semantic Mapping:** Intelligently maps execution phases to ensure report consistency across both Deep Learning and Machine Learning frameworks:
    * **`fit()`** operations are reported as **Model (Backward)** to represent the training and weight-update phase;
    * **`predict()`** and **`transform()`** operations are reported as **Model (Forward)** to represent the inference phase;
* **Algorithmic Complexity:** Applies targeted mathematical complexity formulas (e.g., $O(n_{trees} \cdot n_{samples} \cdot \log_2(n_{samples}))$ for Random Forests) based on array shapes and data types to provide accurate, hardware-independent workload and BOPs estimates.

---

### 📊 Detailed Reporting

The `FLOPpyReport` object provides a detailed, phase-aware breakdown of the computational workload:

* **`model_forward_flops` & `model_forward_bops`**: The algorithmic cost and precision-aware hardware effort (Bit-Operations) of the forward pass. In Scikit-learn workflows, this maps to inference methods like `predict()` and `transform()`;
* **`model_backward_flops` & `model_backward_bops`**: The computational workload required for the training phase. This captures the Autograd gradient calculation in Deep Learning, or the `fit()` method in classical Machine Learning;
* **`loss_forward_flops` & `loss_forward_bops`**: The operations and actual hardware effort explicitly tied to evaluating the loss function;
* **`optimizer_flops` & `optimizer_bops`**: The computational overhead of the optimization step (e.g., weight updates, momentum). It accounts for the specific bit-width used, accurately tracking even fused or quantized optimizers (e.g., 8-bit Adam) via the built-in *Escape Hatch*;
* **`preproc_ops`**: Workload from input preparation, such as tokenizer operations for Large Language Models;
* **`System Environment`**: A detailed snapshot of the execution context, including CPU/GPU specifications, RAM, OS, and active library versions (e.g., PyTorch, Scikit-learn).
---

## ✍️ Authors & Citation

**Francesco Scala, Francesco Mandarino, Liliana Martirano, and Luigi Pontieri.** *Institute of High Performance Computing and Networking (ICAR-CNR) & University of Calabria, Italy.*

If you use FLOPpy in your research, please cite:

 ```
  @article{SCALA2026102865,
   title = {FLOPpy: A hardware-agnostic Python library to monitor the computational cost of machine and deep learning algorithms},
    journal = {SoftwareX},
    volume = {35},
    pages = {102865},
    year = {2026},
    issn = {2352-7110},
    doi = {https://doi.org/10.1016/j.softx.2026.102865},
    url = {https://www.sciencedirect.com/science/article/pii/S2352711026003572},
    author = {Francesco Scala and Francesco Mandarino and Liliana Martirano and Luigi Pontieri},
    keywords = {Green AI, Computational workload, Hardware-agnostic, Deep learning, Machine learning, Floating point operations, Bit-operations}
}
```

---

## 📄 License

This software is licensed under the **GNU Public License v3.0 (GPL3)**.
