#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_dot_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_mv_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_mm_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_bmm_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_baddbmm_flops(
    const at::RecordFunction& function
);

std::uint64_t calculate_addmm_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
