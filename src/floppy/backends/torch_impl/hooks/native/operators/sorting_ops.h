#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Computes the legacy FLOPpy cost for topk, sort, and argsort.
 *
 * numel(input) * log2(max(2, numel(input)))
 */
std::uint64_t calculate_sorting_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
