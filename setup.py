from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "floppy.backends.torch_impl.hooks._flop_counter_core",
        ["src/floppy/backends/torch_impl/hooks/flop_counter_core.cpp"],
        cxx_std=17,
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)