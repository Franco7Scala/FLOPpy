#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Legacy FLOPpy pooling cost:
 *
 * FLOP = number of output elements.
 */
std::uint64_t calculate_pooling_flops(
    const at::RecordFunction& function
);

/**
 * Legacy FLOPpy cost for adaptive average pooling:
 *
 * FLOP = number of output elements.
 */
std::uint64_t calculate_adaptive_avg_pooling_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
