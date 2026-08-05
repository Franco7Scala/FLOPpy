#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Computes the FLOPs for the canonical aten::_convolution operator.
 *
 * Supports standard and transposed 1D, 2D, and 3D convolutions, including grouped and depthwise variants
 */
std::uint64_t calculate_convolution_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
