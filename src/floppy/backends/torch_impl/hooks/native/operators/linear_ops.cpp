#include "linear_ops.h"

#include "../cost_utils.h"

namespace floppy::native {

/**
 * Calcola i FLOP di aten::dot.
 *
 * Operazione:
 *
 * output = dot(first, second)
 *
 * first  = [K]
 * second = [K]
 *
 * Convenzione:
 *
 * FLOP = 2 * K
 *
 * Utilizziamo la stessa convenzione adottata per mm e bmm:
 * una moltiplicazione e un'addizione per ogni elemento.
 */
std::uint64_t calculate_dot_flops(
    const at::RecordFunction& function
) {
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

    const at::Tensor first =
        inputs[0].toTensor();

    const at::Tensor second =
        inputs[1].toTensor();

    if (
        !first.defined() ||
        !second.defined()
    ) {
        return 0;
    }

    if (
        first.dim() != 1 ||
        second.dim() != 1
    ) {
        return 0;
    }

    const int64_t first_size_signed =
        first.size(0);

    const int64_t second_size_signed =
        second.size(0);

    if (
        first_size_signed !=
        second_size_signed
    ) {
        return 0;
    }

    const std::uint64_t vector_size =
        dimension_to_uint64(
            first_size_signed
        );

    return checked_multiply(
        2,
        vector_size
    );
}


/**
 * Calcola i FLOP di aten::mv.
 *
 * Operazione:
 *
 * output = matrix @ vector
 *
 * matrix = [M, K]
 * vector = [K]
 * output = [M]
 *
 * Convenzione:
 *
 * FLOP = 2 * M * K
 */
std::uint64_t calculate_mv_flops(
    const at::RecordFunction& function
) {
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

    const at::Tensor matrix =
        inputs[0].toTensor();

    const at::Tensor vector =
        inputs[1].toTensor();

    if (
        !matrix.defined() ||
        !vector.defined()
    ) {
        return 0;
    }

    if (
        matrix.dim() != 2 ||
        vector.dim() != 1
    ) {
        return 0;
    }

    const int64_t rows_signed =
        matrix.size(0);

    const int64_t inner_dimension_signed =
        matrix.size(1);

    const int64_t vector_size_signed =
        vector.size(0);

    if (
        inner_dimension_signed !=
        vector_size_signed
    ) {
        return 0;
    }

    const std::uint64_t rows =
        dimension_to_uint64(
            rows_signed
        );

    const std::uint64_t inner_dimension =
        dimension_to_uint64(
            inner_dimension_signed
        );

    return checked_multiply(
        checked_multiply(
            2,
            rows
        ),
        inner_dimension
    );
}


/**
 * Calcola i FLOP di aten::mm.
 *
 * Operazione:
 *
 * output = first @ second
 *
 * first  = [M, K]
 * second = [K, N]
 *
 * Convenzione:
 *
 * FLOP = 2 * M * N * K
 */
std::uint64_t calculate_mm_flops(
    const at::RecordFunction& function
) {
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

    const at::Tensor first =
        inputs[0].toTensor();

    const at::Tensor second =
        inputs[1].toTensor();

    if (
        !first.defined() ||
        !second.defined()
    ) {
        return 0;
    }

    if (
        first.dim() != 2 ||
        second.dim() != 2
    ) {
        return 0;
    }

    const int64_t m_signed =
        first.size(0);

    const int64_t first_k_signed =
        first.size(1);

    const int64_t second_k_signed =
        second.size(0);

    const int64_t n_signed =
        second.size(1);

    if (first_k_signed != second_k_signed) {
        return 0;
    }

    const std::uint64_t m =
        dimension_to_uint64(m_signed);

    const std::uint64_t k =
        dimension_to_uint64(first_k_signed);

    const std::uint64_t n =
        dimension_to_uint64(n_signed);

    std::uint64_t flops = 2;

    flops = checked_multiply(flops, m);
    flops = checked_multiply(flops, n);
    flops = checked_multiply(flops, k);

    return flops;
}


/**
 * Calcola i FLOP di aten::bmm.
 *
 * Operazione:
 *
 * output = first @ second
 *
 * first  = [B, M, K]
 * second = [B, K, N]
 *
 * Convenzione:
 *
 * FLOP = 2 * B * M * N * K
 */
std::uint64_t calculate_bmm_flops(
    const at::RecordFunction& function
) {
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

    const at::Tensor first =
        inputs[0].toTensor();

    const at::Tensor second =
        inputs[1].toTensor();

    if (
        !first.defined() ||
        !second.defined()
    ) {
        return 0;
    }

    if (
        first.dim() != 3 ||
        second.dim() != 3
    ) {
        return 0;
    }

    const int64_t first_batch_signed =
        first.size(0);

    const int64_t second_batch_signed =
        second.size(0);

    const int64_t m_signed =
        first.size(1);

    const int64_t first_k_signed =
        first.size(2);

    const int64_t second_k_signed =
        second.size(1);

    const int64_t n_signed =
        second.size(2);

    if (
        first_batch_signed !=
        second_batch_signed
    ) {
        return 0;
    }

    if (first_k_signed != second_k_signed) {
        return 0;
    }

    const std::uint64_t batch =
        dimension_to_uint64(
            first_batch_signed
        );

    const std::uint64_t m =
        dimension_to_uint64(m_signed);

    const std::uint64_t k =
        dimension_to_uint64(first_k_signed);

    const std::uint64_t n =
        dimension_to_uint64(n_signed);

    std::uint64_t flops = 2;

    flops =
        checked_multiply(flops, batch);

    flops =
        checked_multiply(flops, m);

    flops =
        checked_multiply(flops, n);

    flops =
        checked_multiply(flops, k);

    return flops;
}


/**
 * Calcola i FLOP di aten::addmm.
 *
 * Operazione:
 *
 * output = beta * self + alpha * (mat1 @ mat2)
 *
 * Convenzione adottata:
 *
 * parte matriciale = 2 * M * N * K
 * somma del bias   = M * N
 *
 * Le moltiplicazioni per alpha e beta non vengono
 * conteggiate separatamente.
 */
std::uint64_t calculate_addmm_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 3) {
        return 0;
    }

    if (
        !inputs[0].isTensor() ||
        !inputs[1].isTensor() ||
        !inputs[2].isTensor()
    ) {
        return 0;
    }

    const at::Tensor self =
        inputs[0].toTensor();

    const at::Tensor mat1 =
        inputs[1].toTensor();

    const at::Tensor mat2 =
        inputs[2].toTensor();

    if (
        !mat1.defined() ||
        !mat2.defined()
    ) {
        return 0;
    }

    if (
        mat1.dim() != 2 ||
        mat2.dim() != 2
    ) {
        return 0;
    }

    const int64_t m_signed =
        mat1.size(0);

    const int64_t first_k_signed =
        mat1.size(1);

    const int64_t second_k_signed =
        mat2.size(0);

    const int64_t n_signed =
        mat2.size(1);

    if (first_k_signed != second_k_signed) {
        return 0;
    }

    const std::uint64_t m =
        dimension_to_uint64(m_signed);

    const std::uint64_t k =
        dimension_to_uint64(first_k_signed);

    const std::uint64_t n =
        dimension_to_uint64(n_signed);

    std::uint64_t matrix_flops = 2;

    matrix_flops =
        checked_multiply(matrix_flops, m);

    matrix_flops =
        checked_multiply(matrix_flops, n);

    matrix_flops =
        checked_multiply(matrix_flops, k);

    std::uint64_t addition_flops = 0;

    if (
        self.defined() &&
        tensor_numel(self) > 0
    ) {
        addition_flops =
            checked_multiply(m, n);
    }

    return checked_add(
        matrix_flops,
        addition_flops
    );
}

}  // namespace floppy::native
