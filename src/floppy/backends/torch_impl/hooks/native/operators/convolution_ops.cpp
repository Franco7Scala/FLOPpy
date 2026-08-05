#include "convolution_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace floppy::native {

namespace {

std::uint64_t calculate_standard_output_dimension(
    const int64_t input_size,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation
) {
    if (
        input_size < 0 ||
        kernel_size <= 0 ||
        stride <= 0 ||
        padding < 0 ||
        dilation <= 0
    ) {
        throw std::runtime_error(
            "Invalid convolution dimension parameters."
        );
    }

    const int64_t numerator =
        input_size
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1;

    if (numerator < 0) {
        return 0;
    }

    return dimension_to_uint64(
        numerator / stride + 1
    );
}

std::uint64_t calculate_transposed_output_dimension(
    const int64_t input_size,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const int64_t output_padding
) {
    if (
        input_size < 0 ||
        kernel_size <= 0 ||
        stride <= 0 ||
        padding < 0 ||
        dilation <= 0 ||
        output_padding < 0 ||
        output_padding >= stride
    ) {
        throw std::runtime_error(
            "Invalid transposed convolution dimension parameters."
        );
    }

    const int64_t output_size =
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1;

    if (output_size < 0) {
        return 0;
    }

    return dimension_to_uint64(output_size);
}

bool has_bias_input(
    const c10::ArrayRef<const c10::IValue> inputs
) {
    if (
        inputs.size() < 3 ||
        !inputs[2].isTensor()
    ) {
        return false;
    }

    const at::Tensor bias =
        inputs[2].toTensor();

    return (
        bias.defined() &&
        tensor_numel(bias) > 0
    );
}

}  // namespace

std::uint64_t calculate_convolution_flops(
    const at::RecordFunction& function
) {
    /*
     * Canonical signature of aten::_convolution:
     *
     * 0: input
     * 1: weight
     * 2: bias oppure None
     * 3: stride
     * 4: padding
     * 5: dilation
     * 6: transposed
     * 7: output_padding
     * 8: groups
     */
    const auto inputs = function.inputs();

    if (inputs.size() < 9) {
        return 0;
    }

    if (
        !inputs[0].isTensor() ||
        !inputs[1].isTensor() ||
        !inputs[3].isIntList() ||
        !inputs[4].isIntList() ||
        !inputs[5].isIntList() ||
        !inputs[6].isBool() ||
        !inputs[7].isIntList() ||
        !inputs[8].isInt()
    ) {
        return 0;
    }

    const at::Tensor input =
        inputs[0].toTensor();

    const at::Tensor weight =
        inputs[1].toTensor();

    if (
        !input.defined() ||
        !weight.defined()
    ) {
        return 0;
    }

    if (
        input.dim() < 3 ||
        input.dim() > 5 ||
        weight.dim() != input.dim()
    ) {
        return 0;
    }

    const std::vector<int64_t> stride =
        inputs[3].toIntVector();

    const std::vector<int64_t> padding =
        inputs[4].toIntVector();

    const std::vector<int64_t> dilation =
        inputs[5].toIntVector();

    const bool transposed =
        inputs[6].toBool();

    const std::vector<int64_t> output_padding =
        inputs[7].toIntVector();

    const int64_t groups_signed =
        inputs[8].toInt();

    const std::size_t spatial_dimensions =
        static_cast<std::size_t>(
            input.dim() - 2
        );

    if (
        stride.size() != spatial_dimensions ||
        padding.size() != spatial_dimensions ||
        dilation.size() != spatial_dimensions ||
        output_padding.size() != spatial_dimensions ||
        groups_signed <= 0
    ) {
        return 0;
    }

    const std::uint64_t groups =
        dimension_to_uint64(groups_signed);

    const std::uint64_t batch =
        dimension_to_uint64(input.size(0));

    const std::uint64_t input_channels =
        dimension_to_uint64(input.size(1));

    std::uint64_t kernel_elements = 1;
    std::uint64_t input_spatial_elements = 1;
    std::uint64_t output_spatial_elements = 1;

    for (
        std::size_t dimension = 0;
        dimension < spatial_dimensions;
        ++dimension
    ) {
        const int64_t tensor_dimension =
            static_cast<int64_t>(
                dimension + 2
            );

        const int64_t input_size =
            input.size(tensor_dimension);

        const int64_t kernel_size =
            weight.size(tensor_dimension);

        kernel_elements =
            checked_multiply(
                kernel_elements,
                dimension_to_uint64(
                    kernel_size
                )
            );

        input_spatial_elements =
            checked_multiply(
                input_spatial_elements,
                dimension_to_uint64(
                    input_size
                )
            );

        const std::uint64_t output_size =
            transposed
                ? calculate_transposed_output_dimension(
                    input_size,
                    kernel_size,
                    stride[dimension],
                    padding[dimension],
                    dilation[dimension],
                    output_padding[dimension]
                )
                : calculate_standard_output_dimension(
                    input_size,
                    kernel_size,
                    stride[dimension],
                    padding[dimension],
                    dilation[dimension]
                );

        output_spatial_elements =
            checked_multiply(
                output_spatial_elements,
                output_size
            );
    }

    std::uint64_t output_channels = 0;
    std::uint64_t multiplication_flops = 0;

    if (!transposed) {
        /*
         * Standard weight:
         * [out_channels, in_channels / groups, kernel...]
         */
        const std::uint64_t
            input_channels_per_group =
                dimension_to_uint64(
                    weight.size(1)
                );

        if (
            checked_multiply(
                input_channels_per_group,
                groups
            ) != input_channels
        ) {
            return 0;
        }

        output_channels =
            dimension_to_uint64(
                weight.size(0)
            );

        std::uint64_t output_elements =
            checked_multiply(
                batch,
                output_channels
            );

        output_elements =
            checked_multiply(
                output_elements,
                output_spatial_elements
            );

        std::uint64_t operations_per_output =
            checked_multiply(
                2,
                input_channels_per_group
            );

        operations_per_output =
            checked_multiply(
                operations_per_output,
                kernel_elements
            );

        multiplication_flops =
            checked_multiply(
                output_elements,
                operations_per_output
            );
    }
    else {
        /*
         * Transposed weight:
         * [in_channels, out_channels / groups, kernel...]
         */
        const std::uint64_t
            weight_input_channels =
                dimension_to_uint64(
                    weight.size(0)
                );

        if (weight_input_channels != input_channels) {
            return 0;
        }

        const std::uint64_t
            output_channels_per_group =
                dimension_to_uint64(
                    weight.size(1)
                );

        output_channels =
            checked_multiply(
                output_channels_per_group,
                groups
            );

        std::uint64_t input_elements =
            checked_multiply(
                batch,
                input_channels
            );

        input_elements =
            checked_multiply(
                input_elements,
                input_spatial_elements
            );

        std::uint64_t operations_per_input =
            checked_multiply(
                2,
                output_channels_per_group
            );

        operations_per_input =
            checked_multiply(
                operations_per_input,
                kernel_elements
            );

        multiplication_flops =
            checked_multiply(
                input_elements,
                operations_per_input
            );
    }

    if (!has_bias_input(inputs)) {
        return multiplication_flops;
    }

    std::uint64_t output_elements =
        checked_multiply(
            batch,
            output_channels
        );

    output_elements =
        checked_multiply(
            output_elements,
            output_spatial_elements
        );

    return checked_add(
        multiplication_flops,
        output_elements
    );
}

}  // namespace floppy::native
