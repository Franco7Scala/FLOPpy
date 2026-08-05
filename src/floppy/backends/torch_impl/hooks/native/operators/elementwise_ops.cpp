#include "elementwise_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace floppy::native {

namespace {

std::uint64_t broadcast_numel(
    const at::Tensor& first,
    const at::Tensor& second
) {
    if (!first.defined() || !second.defined()) {
        return 0;
    }

    const auto first_sizes = first.sizes();
    const auto second_sizes = second.sizes();

    const std::size_t first_rank =
        static_cast<std::size_t>(first_sizes.size());
    const std::size_t second_rank =
        static_cast<std::size_t>(second_sizes.size());
    const std::size_t output_rank =
        std::max(first_rank, second_rank);

    std::uint64_t elements = 1;

    for (std::size_t offset = 0; offset < output_rank; ++offset) {
        const int64_t first_dimension =
            offset < first_rank
                ? first_sizes[first_rank - 1 - offset]
                : 1;

        const int64_t second_dimension =
            offset < second_rank
                ? second_sizes[second_rank - 1 - offset]
                : 1;

        if (
            first_dimension != second_dimension &&
            first_dimension != 1 &&
            second_dimension != 1
        ) {
            return 0;
        }

        elements = checked_multiply(
            elements,
            dimension_to_uint64(
                std::max(first_dimension, second_dimension)
            )
        );
    }

    return elements;
}

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

std::uint64_t first_tensor_list_numel(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.empty() || !inputs[0].isTensorList()) {
        return 0;
    }

    std::uint64_t total = 0;

    for (const at::Tensor& tensor : inputs[0].toTensorVector()) {
        if (!tensor.defined()) {
            return 0;
        }

        total = checked_add(total, tensor_numel(tensor));
    }

    return total;
}

}  // namespace

std::uint64_t calculate_unary_elementwise_flops(
    const at::RecordFunction& function
) {
    const at::Tensor input = first_tensor_input(function);
    return input.defined() ? tensor_numel(input) : 0;
}

std::uint64_t calculate_binary_elementwise_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.empty() || !inputs[0].isTensor()) {
        return 0;
    }

    const at::Tensor first = inputs[0].toTensor();

    if (!first.defined()) {
        return 0;
    }

    if (inputs.size() >= 2 && inputs[1].isTensor()) {
        const at::Tensor second = inputs[1].toTensor();
        return second.defined() ? broadcast_numel(first, second) : 0;
    }

    return tensor_numel(first);
}

std::uint64_t calculate_abs_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() != 1 || !inputs[0].isTensor()) {
        return 0;
    }

    const at::Tensor input = inputs[0].toTensor();
    return input.defined() ? tensor_numel(input) : 0;
}

std::uint64_t calculate_add_inplace_flops(
    const at::RecordFunction& function
) {
    return calculate_inplace_elementwise_flops(function);
}

std::uint64_t calculate_inplace_elementwise_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.empty() || !inputs[0].isTensor()) {
        return 0;
    }

    const at::Tensor self = inputs[0].toTensor();
    return self.defined() ? tensor_numel(self) : 0;
}

std::uint64_t calculate_complex_elementwise_flops(
    const at::RecordFunction& function
) {
    return checked_multiply(
        calculate_inplace_elementwise_flops(function),
        3
    );
}

std::uint64_t calculate_foreach_elementwise_flops(
    const at::RecordFunction& function
) {
    return first_tensor_list_numel(function);
}

std::uint64_t calculate_foreach_complex_flops(
    const at::RecordFunction& function
) {
    return checked_multiply(first_tensor_list_numel(function), 3);
}

}  // namespace floppy::native
