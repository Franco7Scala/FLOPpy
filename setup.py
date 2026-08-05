from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


NATIVE_ROOT = "src/floppy/backends/torch_impl/hooks/native"


ext_modules = [
    CppExtension(
        name="floppy.backends.torch_impl.hooks._native_counter_probe",
        sources=[
            f"{NATIVE_ROOT}/bindings.cpp",
            f"{NATIVE_ROOT}/native_counter.cpp",
            f"{NATIVE_ROOT}/operator_registry.cpp",
            f"{NATIVE_ROOT}/dtype_utils.cpp",
            f"{NATIVE_ROOT}/diagnostics.cpp",
            f"{NATIVE_ROOT}/operators/elementwise_ops.cpp",
            f"{NATIVE_ROOT}/operators/optimizer_ops.cpp",
            f"{NATIVE_ROOT}/operators/normalization_ops.cpp",
            f"{NATIVE_ROOT}/operators/embedding_ops.cpp",
            f"{NATIVE_ROOT}/operators/attention_ops.cpp",
            f"{NATIVE_ROOT}/operators/indexing_ops.cpp",
            f"{NATIVE_ROOT}/operators/legacy_compat_ops.cpp",
            f"{NATIVE_ROOT}/operators/pooling_ops.cpp",
            f"{NATIVE_ROOT}/operators/upsampling_ops.cpp",
            f"{NATIVE_ROOT}/operators/sorting_ops.cpp",
            f"{NATIVE_ROOT}/operators/recurrent_ops.cpp",
            f"{NATIVE_ROOT}/operators/linear_ops.cpp",
            f"{NATIVE_ROOT}/operators/loss_ops.cpp",
            f"{NATIVE_ROOT}/operators/convolution_ops.cpp",
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
            ],
        },
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension,
    },
)
