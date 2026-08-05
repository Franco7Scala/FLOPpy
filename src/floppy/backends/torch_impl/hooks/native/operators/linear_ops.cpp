#include "linear_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstdint>

namespace floppy::native {

namespace {

std::uint64_t matrix_product_flops(
    const std::uint64_t batch,
    const std::uint64_t rows,
    const std::uint64_t columns,
    const std::uint64_t inner_dimension
) {
    std::uint64_t flops = 2;
    flops = checked_multiply(flops, batch);
    flops = checked_multiply(flops, rows);
    flops = checked_multiply(flops, columns);
    flops = checked_multiply(flops, inner_dimension);
    return flops;
}

}  // namespace

std::uint64_t calculate_dot_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 2 || !inputs[0].isTensor() || !inputs[1].isTensor()) {
        return 0;
    }

    const at::Tensor first = inputs[0].toTensor();
    const at::Tensor second = inputs[1].toTensor();

    if (
        !first.defined() ||
        !second.defined() ||
        first.dim() != 1 ||
        second.dim() != 1 ||
        first.size(0) != second.size(0)
    ) {
        return 0;
    }

    return checked_multiply(2, dimension_to_uint64(first.size(0)));
}

std::uint64_t calculate_mv_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 2 || !inputs[0].isTensor() || !inputs[1].isTensor()) {
        return 0;
    }

    const at::Tensor matrix = inputs[0].toTensor();
    const at::Tensor vector = inputs[1].toTensor();

    if (
        !matrix.defined() ||
        !vector.defined() ||
        matrix.dim() != 2 ||
        vector.dim() != 1 ||
        matrix.size(1) != vector.size(0)
    ) {
        return 0;
    }

    return matrix_product_flops(
        1,
        dimension_to_uint64(matrix.size(0)),
        1,
        dimension_to_uint64(matrix.size(1))
    );
}

std::uint64_t calculate_mm_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 2 || !inputs[0].isTensor() || !inputs[1].isTensor()) {
        return 0;
    }

    const at::Tensor first = inputs[0].toTensor();
    const at::Tensor second = inputs[1].toTensor();

    if (
        !first.defined() ||
        !second.defined() ||
        first.dim() != 2 ||
        second.dim() != 2 ||
        first.size(1) != second.size(0)
    ) {
        return 0;
    }

    return matrix_product_flops(
        1,
        dimension_to_uint64(first.size(0)),
        dimension_to_uint64(second.size(1)),
        dimension_to_uint64(first.size(1))
    );
}

std::uint64_t calculate_bmm_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 2 || !inputs[0].isTensor() || !inputs[1].isTensor()) {
        return 0;
    }

    const at::Tensor first = inputs[0].toTensor();
    const at::Tensor second = inputs[1].toTensor();

    if (
        !first.defined() ||
        !second.defined() ||
        first.dim() != 3 ||
        second.dim() != 3 ||
        first.size(0) != second.size(0) ||
        first.size(2) != second.size(1)
    ) {
        return 0;
    }

    return matrix_product_flops(
        dimension_to_uint64(first.size(0)),
        dimension_to_uint64(first.size(1)),
        dimension_to_uint64(second.size(2)),
        dimension_to_uint64(first.size(2))
    );
}

std::uint64_t calculate_baddbmm_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 3 || !inputs[1].isTensor() || !inputs[2].isTensor()) {
        return 0;
    }

    const at::Tensor batch1 = inputs[1].toTensor();
    const at::Tensor batch2 = inputs[2].toTensor();

    if (
        !batch1.defined() ||
        !batch2.defined() ||
        batch1.dim() != 3 ||
        batch2.dim() != 3 ||
        batch1.size(0) != batch2.size(0) ||
        batch1.size(2) != batch2.size(1)
    ) {
        return 0;
    }

    return matrix_product_flops(
        dimension_to_uint64(batch1.size(0)),
        dimension_to_uint64(batch1.size(1)),
        dimension_to_uint64(batch2.size(2)),
        dimension_to_uint64(batch1.size(2))
    );
}

std::uint64_t calculate_addmm_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (
        inputs.size() < 3 ||
        !inputs[0].isTensor() ||
        !inputs[1].isTensor() ||
        !inputs[2].isTensor()
    ) {
        return 0;
    }

    const at::Tensor self = inputs[0].toTensor();
    const at::Tensor mat1 = inputs[1].toTensor();
    const at::Tensor mat2 = inputs[2].toTensor();

    if (
        !mat1.defined() ||
        !mat2.defined() ||
        mat1.dim() != 2 ||
        mat2.dim() != 2 ||
        mat1.size(1) != mat2.size(0)
    ) {
        return 0;
    }

    const std::uint64_t rows = dimension_to_uint64(mat1.size(0));
    const std::uint64_t columns = dimension_to_uint64(mat2.size(1));

    const std::uint64_t matrix_flops = matrix_product_flops(
        1,
        rows,
        columns,
        dimension_to_uint64(mat1.size(1))
    );

    const std::uint64_t addition_flops =
        self.defined() && tensor_numel(self) > 0
            ? checked_multiply(rows, columns)
            : 0;

    return checked_add(matrix_flops, addition_flops);
}

}  // namespace floppy::native
