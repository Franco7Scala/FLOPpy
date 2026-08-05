#include "embedding_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_embedding_flops(
    const at::RecordFunction& function
) {
    /*
     * Firma di aten::embedding:
     *
     * 0: weight
     * 1: indices
     * 2: padding_idx
     * 3: scale_grad_by_freq
     * 4: sparse
     */
    const auto inputs = function.inputs();

    if (inputs.size() < 2) {
        return 0;
    }

    if (
        !inputs[0].isTensor() ||
        !inputs[1].isTensor()
    ) {
        return 0;
    }

    const at::Tensor weight =
        inputs[0].toTensor();

    const at::Tensor indices =
        inputs[1].toTensor();

    if (
        !weight.defined() ||
        !indices.defined() ||
        weight.dim() < 2
    ) {
        return 0;
    }

    const std::uint64_t index_count =
        tensor_numel(indices);

    const std::uint64_t embedding_dimension =
        dimension_to_uint64(
            weight.size(-1)
        );

    return checked_multiply(
        index_count,
        embedding_dimension
    );
}

}  // namespace floppy::native
