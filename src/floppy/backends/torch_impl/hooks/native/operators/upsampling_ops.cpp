#include "upsampling_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace floppy::native {

std::uint64_t calculate_upsampling_flops(
    const at::RecordFunction& function
) {
    /*
     * Common observed signature:
     *
     * 0: input
     * 1: output_size
     * 2+: mode-specific parameters
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

    return checked_multiply(
        output_elements,
        4
    );
}

}  // namespace floppy::native
