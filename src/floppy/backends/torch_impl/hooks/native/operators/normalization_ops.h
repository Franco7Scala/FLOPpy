#pragma once
#include <ATen/record_function.h>
#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_native_layer_norm_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_softmax_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
