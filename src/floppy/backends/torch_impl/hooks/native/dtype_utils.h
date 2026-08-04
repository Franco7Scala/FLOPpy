#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t tensor_bit_width(const at::Tensor& tensor);

std::uint64_t get_effective_bit_width(
    const at::RecordFunction& function
);

}  // namespace floppy::native
