#include "loss_ops.h"

#include "../cost_utils.h"
#include <ATen/ATen.h>

namespace floppy::native {

std::uint64_t calculate_mse_loss_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 3) {
        return 0;
    }

    if (
        !inputs[0].isTensor() ||
        !inputs[1].isTensor() ||
        !inputs[2].isInt()
    ) {
        return 0;
    }

    const at::Tensor input = inputs[0].toTensor();
    const at::Tensor target = inputs[1].toTensor();

    if (
        !input.defined() ||
        !target.defined() ||
        input.sizes() != target.sizes()
    ) {
        return 0;
    }

    const std::uint64_t elements = tensor_numel(input);
    const int64_t reduction = inputs[2].toInt();

    std::uint64_t flops = checked_multiply(elements, 2);

    if (reduction == 0) {
        return flops;
    }

    if (reduction == 1 || reduction == 2) {
        flops = checked_add(flops, elements);
    }
    else {
        return 0;
    }

    if (reduction == 1) {
        flops = checked_add(flops, 1);
    }

    return flops;
}

std::uint64_t calculate_nll_loss_forward_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (
        inputs.empty() ||
        !inputs[0].isTensor()
    ) {
        return 0;
    }

    const at::Tensor input = inputs[0].toTensor();

    if (!input.defined()) {
        return 0;
    }

    return tensor_numel(input);
}

}  // namespace floppy::native
