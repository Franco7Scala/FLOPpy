#include "indexing_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>

namespace floppy::native {

namespace {

at::Tensor tensor_input_at(
    const at::RecordFunction& function,
    const std::size_t index
) {
    const auto inputs = function.inputs();

    if (index >= inputs.size() || !inputs[index].isTensor()) {
        return {};
    }

    const at::Tensor tensor = inputs[index].toTensor();
    return tensor.defined() ? tensor : at::Tensor{};
}

}  // namespace

std::uint64_t calculate_destination_indexing_flops(
    const at::RecordFunction& function
) {
    const at::Tensor destination = tensor_input_at(function, 0);
    return destination.defined() ? tensor_numel(destination) : 0;
}

std::uint64_t calculate_index_select_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (
        inputs.size() < 3 ||
        !inputs[0].isTensor() ||
        !inputs[1].isInt() ||
        !inputs[2].isTensor()
    ) {
        return 0;
    }

    const at::Tensor input = inputs[0].toTensor();
    const at::Tensor index = inputs[2].toTensor();

    if (!input.defined() || !index.defined() || input.dim() <= 0) {
        return 0;
    }

    int64_t dimension = inputs[1].toInt();

    if (dimension < 0) {
        dimension += input.dim();
    }

    if (dimension < 0 || dimension >= input.dim()) {
        return 0;
    }

    std::uint64_t output_elements = tensor_numel(index);

    for (int64_t current = 0; current < input.dim(); ++current) {
        if (current == dimension) {
            continue;
        }

        output_elements = checked_multiply(
            output_elements,
            dimension_to_uint64(input.size(current))
        );
    }

    return output_elements;
}

std::uint64_t calculate_gather_flops(
    const at::RecordFunction& function
) {
    const at::Tensor index = tensor_input_at(function, 2);
    return index.defined() ? tensor_numel(index) : 0;
}

std::uint64_t calculate_bincount_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input = tensor_input_at(function, 0);
    return input.defined() ? tensor_numel(input) : 0;
}

}  // namespace floppy::native
