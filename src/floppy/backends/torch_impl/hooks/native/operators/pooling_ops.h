#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Costo legacy FLOPpy per pooling:
 *
 * FLOP = numero di elementi dell'output.
 */
std::uint64_t calculate_pooling_flops(
    const at::RecordFunction& function
);

/**
 * Costo legacy FLOPpy per adaptive average pooling:
 *
 * FLOP = numero di elementi dell'output.
 */
std::uint64_t calculate_adaptive_avg_pooling_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
