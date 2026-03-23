.. floppy-lib documentation master file

FLOPpy: Hardware-Agnostic Workload Monitoring
=============================================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

**FLOPpy** is a Python library designed to estimate and monitor the computational workload of Machine Learning (ML) and Deep Learning (DL) models. By systematically counting **Floating Point Operations (FLOPs)**, the library enables a hardware-independent assessment of the algorithmic workload. Unlike metrics such as execution time or energy consumption, FLOPs reflect the intrinsic computational demand of an algorithm, ensuring comparability and reproducibility across different systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   methodology
   metrics
   api_reference

🚀 Key Features
---------------

* **Hardware-Agnostic Monitoring**: Provides a standardized measure of computational demand that does not depend on specific hardware characteristics or infrastructure.
* **Broad Framework Support**: Offers a unified interface for models implemented in **PyTorch** (including also **Hugging Face** models) and **Scikit-learn**.
* **Comprehensive Pipeline Tracking**: Beyond simple forward passes, it accounts for **backward passes**, **optimizer updates**, **loss evaluations**, **tokenizer operations** and so on.
* **Modular Architecture**: Designed with a provider pattern and structural decoupling, allowing easy extension to other backends.
* **Real-time Integration**: Supports seamless synchronization with **Weights & Biases (WandB)** for real-time visualization.

✍️ Authors & Citation
---------------------

**Francesco Scala, Francesco Mandarino, Liliana Martirano, and Luigi Pontieri.** *Institute of High Performance Computing and Networking (ICAR-CNR) & University of Calabria, Italy.*

If you use FLOPpy in your research, please cite:

   TODO after publication

📄 License
----------

This software is licensed under the **GNU Public License v3.0 (GPL3)**.