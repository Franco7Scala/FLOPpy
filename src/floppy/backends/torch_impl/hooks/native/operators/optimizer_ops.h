#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Operazioni in-place element-wise usate dagli optimizer:
 * un FLOP per elemento.
 *
 * Usata per aten::mul_ e aten::div_.
 */
std::uint64_t calculate_optimizer_unary_inplace_flops(
    const at::RecordFunction& function
);

/**
 * Interpolazione lineare in-place:
 *
 * self = self + weight * (end - self)
 *
 * Convenzione:
 * 2 FLOP per elemento.
 */
std::uint64_t calculate_lerp_inplace_flops(
    const at::RecordFunction& function
);

/**
 * Operazioni composte:
 *
 * addcmul_: self += value * tensor1 * tensor2
 * addcdiv_: self += value * tensor1 / tensor2
 *
 * Convenzione storica FLOPpy:
 * 3 FLOP per elemento.
 */
std::uint64_t calculate_complex_optimizer_inplace_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
