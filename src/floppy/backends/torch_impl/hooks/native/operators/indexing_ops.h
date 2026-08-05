#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_destination_indexing_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_index_select_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_gather_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_bincount_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
