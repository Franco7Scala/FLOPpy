#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Calcola il costo legacy FLOPpy degli operatori di upsampling.
 *
 * Convenzione:
 *
 * FLOP = 4 * numero_elementi_output
 */
std::uint64_t calculate_upsampling_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
