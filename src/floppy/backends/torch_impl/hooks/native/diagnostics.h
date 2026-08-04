#pragma once

#include <ATen/record_function.h>

#include <string>

namespace floppy::native {

std::string describe_inputs(
    const at::RecordFunction& function
);

}  // namespace floppy::native
