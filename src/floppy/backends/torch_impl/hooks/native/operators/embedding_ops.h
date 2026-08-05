#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Computes the historical FLOPpy cost of aten::embedding.
 *
 * Legacy convention:
 *
 * FLOP = number_of_indices * embedding_dimension
 *
 * This is a cost metric maintained for compatibility with the previous implementation, even though
 * embedding is primarily a lookup/memory operation.
 */
std::uint64_t calculate_embedding_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
