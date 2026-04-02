import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "floppy-tracker"
copyright = "2026, Francesco Scala, Francesco Mandarino, Liliana Martirano, Luigi Pontieri"
author = "Francesco Scala, Francesco Mandarino, Liliana Martirano, Luigi Pontieri"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]