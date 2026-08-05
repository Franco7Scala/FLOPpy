#include "normalization_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

namespace floppy::native {

namespace {

at::Tensor first_tensor_input(
    const at::RecordFunction& function
) {
    for (const c10::IValue& value : function.inputs()) {
        if (value.isTensor()) {
            const at::Tensor tensor = value.toTensor();

            if (tensor.defined()) {
                return tensor;
            }
        }
    }

    return {};
}

std::uint64_t scaled_input_numel(
    const at::RecordFunction& function,
    const std::uint64_t scale
) {
    const at::Tensor input = first_tensor_input(function);
    return input.defined()
        ? checked_multiply(tensor_numel(input), scale)
        : 0;
}

}  // namespace

std::uint64_t calculate_native_layer_norm_flops(
    const at::RecordFunction& function
) {
    return scaled_input_numel(function, 4);
}

std::uint64_t calculate_rms_norm_flops(
    const at::RecordFunction& function
) {
    return scaled_input_numel(function, 4);
}

std::uint64_t calculate_softmax_flops(
    const at::RecordFunction& function
) {
    return scaled_input_numel(function, 3);
}

}  // namespace floppy::native
