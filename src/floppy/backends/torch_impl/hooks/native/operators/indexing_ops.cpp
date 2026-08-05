#include "indexing_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstdint>

namespace floppy::native {

namespace {

at::Tensor tensor_input_at(
    const at::RecordFunction& function,
    const std::size_t index
) {
    const auto inputs = function.inputs();

    if (
        index >= inputs.size() ||
        !inputs[index].isTensor()
    ) {
        return {};
    }

    const at::Tensor tensor =
        inputs[index].toTensor();

    if (!tensor.defined()) {
        return {};
    }

    return tensor;
}

}  // namespace

std::uint64_t calculate_destination_indexing_flops(
    const at::RecordFunction& function
) {
    /*
     * scatter, scatter_add, scatter_reduce e index_add
     * restituiscono un tensore con la stessa forma del
     * tensore destinazione, presente in posizione 0.
     */
    const at::Tensor destination =
        tensor_input_at(function, 0);

    if (!destination.defined()) {
        return 0;
    }

    return tensor_numel(destination);
}

std::uint64_t calculate_gather_flops(
    const at::RecordFunction& function
) {
    /*
     * Firma di aten::gather:
     *
     * 0: input
     * 1: dim
     * 2: index
     * 3: sparse_grad
     *
     * L'output ha la stessa forma di index.
     */
    const at::Tensor index =
        tensor_input_at(function, 2);

    if (!index.defined()) {
        return 0;
    }

    return tensor_numel(index);
}

std::uint64_t calculate_bincount_flops(
    const at::RecordFunction& function
) {
    /*
     * Firma di aten::bincount:
     *
     * 0: input
     * 1: weights oppure None
     * 2: minlength
     */
    const at::Tensor input =
        tensor_input_at(function, 0);

    if (!input.defined()) {
        return 0;
    }

    return tensor_numel(input);
}

}  // namespace floppy::native
