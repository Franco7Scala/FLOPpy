#include "pooling_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace floppy::native {

namespace {

std::uint64_t calculate_output_dimension(
    const int64_t input_size,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const bool ceil_mode
) {
    if (
        input_size < 0 ||
        kernel_size <= 0 ||
        stride <= 0 ||
        padding < 0 ||
        dilation <= 0
    ) {
        throw std::runtime_error(
            "Invalid pooling dimension parameters."
        );
    }

    const int64_t effective_kernel =
        dilation * (kernel_size - 1) + 1;

    const int64_t numerator =
        input_size
        + 2 * padding
        - effective_kernel;

    if (numerator < 0) {
        return 0;
    }

    int64_t output_size = 0;

    if (ceil_mode) {
        output_size =
            (numerator + stride - 1) / stride + 1;
    }
    else {
        output_size =
            numerator / stride + 1;
    }

    /*
     * Correzione coerente con il comportamento di PyTorch:
     * l'ultima finestra non deve iniziare interamente nel padding.
     */
    if (
        ceil_mode &&
        (output_size - 1) * stride >= input_size + padding
    ) {
        --output_size;
    }

    if (output_size < 0) {
        return 0;
    }

    return dimension_to_uint64(output_size);
}

std::vector<int64_t> normalized_int_list(
    const c10::IValue& value,
    const std::size_t dimensions,
    const int64_t default_value
) {
    if (!value.isIntList()) {
        return {};
    }

    std::vector<int64_t> values =
        value.toIntVector();

    if (values.empty()) {
        values.assign(
            dimensions,
            default_value
        );
    }
    else if (values.size() == 1 && dimensions > 1) {
        values.assign(
            dimensions,
            values[0]
        );
    }

    return values;
}

}  // namespace

std::uint64_t calculate_pooling_flops(
    const at::RecordFunction& function
) {
    /*
     * Firme coperte:
     *
     * max_pool1d
     * max_pool2d_with_indices
     * max_pool3d_with_indices
     * avg_pool2d
     * avg_pool3d
     *
     * Layout comune:
     * 0: input
     * 1: kernel_size
     * 2: stride
     * 3: padding
     *
     * Per max pool:
     * 4: dilation
     * 5: ceil_mode
     *
     * Per avg pool:
     * 4: ceil_mode
     * 5: count_include_pad
     * 6: divisor_override (2D/3D)
     */
    const auto inputs = function.inputs();

    if (
        inputs.size() < 5 ||
        !inputs[0].isTensor() ||
        !inputs[1].isIntList() ||
        !inputs[2].isIntList() ||
        !inputs[3].isIntList()
    ) {
        return 0;
    }

    const at::Tensor input =
        inputs[0].toTensor();

    if (
        !input.defined() ||
        input.dim() < 3 ||
        input.dim() > 5
    ) {
        return 0;
    }

    const std::size_t spatial_dimensions =
        static_cast<std::size_t>(
            input.dim() - 2
        );

    const std::vector<int64_t> kernel =
        normalized_int_list(
            inputs[1],
            spatial_dimensions,
            1
        );

    std::vector<int64_t> stride =
        normalized_int_list(
            inputs[2],
            spatial_dimensions,
            0
        );

    const std::vector<int64_t> padding =
        normalized_int_list(
            inputs[3],
            spatial_dimensions,
            0
        );

    if (stride.empty()) {
        stride = kernel;
    }

    if (
        kernel.size() != spatial_dimensions ||
        stride.size() != spatial_dimensions ||
        padding.size() != spatial_dimensions
    ) {
        return 0;
    }

    const std::string operator_name =
        function.name();

    const bool is_max_pool =
        operator_name.find("max_pool") !=
        std::string::npos;

    std::vector<int64_t> dilation(
        spatial_dimensions,
        1
    );

    bool ceil_mode = false;

    if (is_max_pool) {
        if (
            inputs.size() < 6 ||
            !inputs[4].isIntList() ||
            !inputs[5].isBool()
        ) {
            return 0;
        }

        dilation =
            normalized_int_list(
                inputs[4],
                spatial_dimensions,
                1
            );

        ceil_mode =
            inputs[5].toBool();
    }
    else {
        if (!inputs[4].isBool()) {
            return 0;
        }

        ceil_mode =
            inputs[4].toBool();
    }

    if (dilation.size() != spatial_dimensions) {
        return 0;
    }

    std::uint64_t output_elements =
        dimension_to_uint64(
            input.size(0)
        );

    output_elements =
        checked_multiply(
            output_elements,
            dimension_to_uint64(
                input.size(1)
            )
        );

    for (
        std::size_t dimension = 0;
        dimension < spatial_dimensions;
        ++dimension
    ) {
        output_elements =
            checked_multiply(
                output_elements,
                calculate_output_dimension(
                    input.size(
                        static_cast<int64_t>(
                            dimension + 2
                        )
                    ),
                    kernel[dimension],
                    stride[dimension],
                    padding[dimension],
                    dilation[dimension],
                    ceil_mode
                )
            );
    }

    return output_elements;
}

std::uint64_t calculate_adaptive_avg_pooling_flops(
    const at::RecordFunction& function
) {
    /*
     * Firma:
     *
     * 0: input
     * 1: output_size
     */
    const auto inputs = function.inputs();

    if (
        inputs.size() < 2 ||
        !inputs[0].isTensor() ||
        !inputs[1].isIntList()
    ) {
        return 0;
    }

    const at::Tensor input =
        inputs[0].toTensor();

    if (
        !input.defined() ||
        input.dim() < 3 ||
        input.dim() > 5
    ) {
        return 0;
    }

    const std::size_t spatial_dimensions =
        static_cast<std::size_t>(
            input.dim() - 2
        );

    const std::vector<int64_t> output_size =
        inputs[1].toIntVector();

    if (output_size.size() != spatial_dimensions) {
        return 0;
    }

    std::uint64_t output_elements =
        dimension_to_uint64(
            input.size(0)
        );

    output_elements =
        checked_multiply(
            output_elements,
            dimension_to_uint64(
                input.size(1)
            )
        );

    for (const int64_t dimension : output_size) {
        output_elements =
            checked_multiply(
                output_elements,
                dimension_to_uint64(
                    dimension
                )
            );
    }

    return output_elements;
}

}  // namespace floppy::native
