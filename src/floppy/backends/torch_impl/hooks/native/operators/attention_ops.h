#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Calculates the FLOPs of aten::_scaled_dot_product_flash_attention.
 *
 * The wrapper aten::scaled_dot_product_attention is not counted, to avoid duplicates.
 *
 * Decomposed math backends are still counted via bmm, softmax, and element-wise operators
 */
std::uint64_t calculate_flash_attention_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
