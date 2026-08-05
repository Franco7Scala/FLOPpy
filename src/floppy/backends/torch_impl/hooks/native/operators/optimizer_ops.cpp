#include "optimizer_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstdint>

namespace floppy::native {

namespace {

at::Tensor first_tensor_input(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    for (const c10::IValue& value : inputs) {
        if (value.isTensor()) {
            const at::Tensor tensor =
                value.toTensor();

            if (tensor.defined()) {
                return tensor;
            }
        }
    }

    return {};
}

}  // namespace

std::uint64_t calculate_optimizer_unary_inplace_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input =
        first_tensor_input(function);

    if (!input.defined()) {
        return 0;
    }

    /*
     * Ignores in-place operations on scalar tensors.
     *
     * PyTorch uses div_ on scalars as an internal implementation detail of certain reductions,
     * such as MSELoss(mean). The reduction cost is already included in the canonical aten::mse_loss
calculator and must not be double counted.
     * Real optimizer operations observed on mul_ and div_, on the other hand, operate on parameter tensors.
     */
    if (input.dim() == 0) {
        return 0;
    }

    return tensor_numel(input);
}

std::uint64_t calculate_lerp_inplace_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input =
        first_tensor_input(function);

    if (!input.defined()) {
        return 0;
    }

    return checked_multiply(
        tensor_numel(input),
        2
    );
}

std::uint64_t calculate_complex_optimizer_inplace_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input =
        first_tensor_input(function);

    if (!input.defined()) {
        return 0;
    }

    return checked_multiply(
        tensor_numel(input),
        3
    );
}

}  // namespace floppy::native
