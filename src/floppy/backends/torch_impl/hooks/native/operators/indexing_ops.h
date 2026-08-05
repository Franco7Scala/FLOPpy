#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Costo legacy per scatter/scatter_add/scatter_reduce/index_add:
 * un FLOP per elemento del tensore destinazione.
 */
std::uint64_t calculate_destination_indexing_flops(
    const at::RecordFunction& function
);

/**
 * Costo legacy per gather:
 * un FLOP per elemento del tensore indice, che coincide
 * con il numero di elementi dell'output.
 */
std::uint64_t calculate_gather_flops(
    const at::RecordFunction& function
);

/**
 * Costo legacy per bincount:
 * un FLOP per elemento del tensore di input.
 */
std::uint64_t calculate_bincount_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
