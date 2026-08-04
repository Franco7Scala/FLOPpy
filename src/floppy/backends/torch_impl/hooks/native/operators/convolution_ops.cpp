#include "convolution_ops.h"

#include "../cost_utils.h"

#include <vector>

namespace floppy::native {

/**
 * Calcola una dimensione spaziale dell'output di una
 * convoluzione non trasposta.
 *
 * Formula:
 *
 * floor(
 *     (
 *         input
 *         + 2 * padding
 *         - dilation * (kernel - 1)
 *         - 1
 *     )
 *     / stride
 *     + 1
 * )
 */
static std::uint64_t
calculate_convolution_output_dimension(
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

    const int64_t output_size =
        numerator / stride + 1;

    return dimension_to_uint64(output_size);
}

/**
 * Calcola i FLOP di aten::mkldnn_convolution.
 *
 * Firma attesa:
 *
 * 0: input
 * 1: weight
 * 2: bias oppure None
 * 3: stride
 * 4: padding
 * 5: dilation
 * 6: groups
 *
 * Formula:
 *
 * 2
 * * numero_elementi_output
 * * canali_input_per_gruppo
 * * numero_elementi_kernel
 *
 * Se è presente un bias viene aggiunta un'operazione
 * per ogni elemento dell'output.
 */
std::uint64_t
calculate_mkldnn_convolution_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 7) {
        return 0;
    }

    if (
        !inputs[0].isTensor() ||
        !inputs[1].isTensor() ||
        !inputs[3].isIntList() ||
        !inputs[4].isIntList() ||
        !inputs[5].isIntList() ||
        !inputs[6].isInt()
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

    const int64_t groups_signed =
        inputs[6].toInt();

    const std::size_t spatial_dimensions =
        static_cast<std::size_t>(
            input.dim() - 2
        );

    if (
        stride.size() != spatial_dimensions ||
        padding.size() != spatial_dimensions ||
        dilation.size() != spatial_dimensions
    ) {
        return 0;
    }

    if (groups_signed <= 0) {
        return 0;
    }

    const std::uint64_t groups =
        dimension_to_uint64(groups_signed);

    const std::uint64_t input_channels =
        dimension_to_uint64(
            input.size(1)
        );

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

    const std::uint64_t batch =
        dimension_to_uint64(
            input.size(0)
        );

    const std::uint64_t output_channels =
        dimension_to_uint64(
            weight.size(0)
        );

    std::uint64_t kernel_elements = 1;
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

        const std::uint64_t output_size =
            calculate_convolution_output_dimension(
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

    std::uint64_t output_elements = batch;

    output_elements =
        checked_multiply(
            output_elements,
            output_channels
        );

    output_elements =
        checked_multiply(
            output_elements,
            output_spatial_elements
        );

    std::uint64_t operations_per_output = 2;

    operations_per_output =
        checked_multiply(
            operations_per_output,
            input_channels_per_group
        );

    operations_per_output =
        checked_multiply(
            operations_per_output,
            kernel_elements
        );

    std::uint64_t convolution_flops =
        checked_multiply(
            output_elements,
            operations_per_output
        );

    bool has_bias = false;

    if (inputs[2].isTensor()) {
        const at::Tensor bias =
            inputs[2].toTensor();

        has_bias =
            bias.defined() &&
            tensor_numel(bias) > 0;
    }

    if (has_bias) {
        convolution_flops =
            checked_add(
                convolution_flops,
                output_elements
            );
    }

    return convolution_flops;
}

}  // namespace floppy::native
