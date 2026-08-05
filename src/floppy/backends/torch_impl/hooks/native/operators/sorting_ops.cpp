#include "sorting_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_sorting_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (
        inputs.empty() ||
        !inputs[0].isTensor()
    ) {
        return 0;
    }

    const at::Tensor input =
        inputs[0].toTensor();

    if (!input.defined()) {
        return 0;
    }

    const std::uint64_t input_elements =
        tensor_numel(input);

    if (input_elements == 0) {
        return 0;
    }

    const double logarithm =
        std::log2(
            static_cast<double>(
                std::max<std::uint64_t>(
                    2,
                    input_elements
                )
            )
        );

    const double estimated_flops =
        static_cast<double>(
            input_elements
        ) * logarithm;

    /*
     * Mantiene la stessa conversione tronca del core legacy.
     */
    return static_cast<std::uint64_t>(
        estimated_flops
    );
}

}  // namespace floppy::native
