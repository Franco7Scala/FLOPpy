#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Calcola il costo legacy FLOPpy per topk, sort e argsort.
 *
 * Formula:
 *
 * numel(input) * log2(max(2, numel(input)))
 */
std::uint64_t calculate_sorting_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
