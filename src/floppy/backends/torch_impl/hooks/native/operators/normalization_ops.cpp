#include "normalization_ops.h"

#include "../cost_utils.h"
#include <ATen/ATen.h>

namespace floppy::native {

namespace {

at::Tensor first_tensor_input(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    for (const c10::IValue& value : inputs) {
        if (value.isTensor()) {
            const at::Tensor tensor = value.toTensor();
            if (tensor.defined()) {
                return tensor;
            }
        }
    }

    return {};
}

}  // namespace

std::uint64_t calculate_native_layer_norm_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input = first_tensor_input(function);

    if (!input.defined()) {
        return 0;
    }

    return checked_multiply(
        tensor_numel(input),
        4
    );
}

std::uint64_t calculate_softmax_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input = first_tensor_input(function);

    if (!input.defined()) {
        return 0;
    }

    return checked_multiply(
        tensor_numel(input),
        3
    );
}

}  // namespace floppy::native
