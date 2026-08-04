#pragma once

#include <pybind11/pybind11.h>

namespace floppy::native {

void bind_native_counter(
    pybind11::module_& module
);

}  // namespace floppy::native
