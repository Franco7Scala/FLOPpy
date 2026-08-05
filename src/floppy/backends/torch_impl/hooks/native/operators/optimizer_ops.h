#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Element-wise in-place operations used by optimizers:
 * one FLOP per element.
 *
 * Used for aten::mul_ and aten::div_.
 */
std::uint64_t calculate_optimizer_unary_inplace_flops(
    const at::RecordFunction& function
);

/**
 * Linear interpolation in-place:
 *
 * self = self + weight * (end - self)
 *
 * Convention:
 * 2 FLOP for element.
 */
std::uint64_t calculate_lerp_inplace_flops(
    const at::RecordFunction& function
);

/**
 * Complex operations:
 *
 * addcmul_: self += value * tensor1 * tensor2
 * addcdiv_: self += value * tensor1 / tensor2
 *
 * Historical FLOPpy convention:
 * 3 FLOP for element.
 */
std::uint64_t calculate_complex_optimizer_inplace_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
