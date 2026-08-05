#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_unary_elementwise_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_binary_elementwise_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_abs_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_add_inplace_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
