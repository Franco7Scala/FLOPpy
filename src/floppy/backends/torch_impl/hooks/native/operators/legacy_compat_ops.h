#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_inplace_elementwise_flops(const at::RecordFunction& function);
std::uint64_t calculate_complex_elementwise_flops(const at::RecordFunction& function);
std::uint64_t calculate_foreach_elementwise_flops(const at::RecordFunction& function);
std::uint64_t calculate_foreach_complex_flops(const at::RecordFunction& function);
std::uint64_t calculate_baddbmm_flops(const at::RecordFunction& function);
std::uint64_t calculate_index_select_flops(const at::RecordFunction& function);
std::uint64_t calculate_rms_norm_flops(const at::RecordFunction& function);

}  // namespace floppy::native
