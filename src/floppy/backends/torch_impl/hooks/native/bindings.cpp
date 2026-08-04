#include "native_counter.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    floppy::native::bind_native_counter(module);
}
