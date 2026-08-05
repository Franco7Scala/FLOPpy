#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Computes the legacy FLOPpy cost for upsampling operations.
 *
 * FLOP = 4 * numero_elementi_output
 */
std::uint64_t calculate_upsampling_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
