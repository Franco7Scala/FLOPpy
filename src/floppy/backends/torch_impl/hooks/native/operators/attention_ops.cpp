#include "attention_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstdint>

namespace floppy::native {

std::uint64_t calculate_flash_attention_flops(
    const at::RecordFunction& function
) {
    /*
     * Firma osservata di
     * aten::_scaled_dot_product_flash_attention:
     *
     * 0: query  [batch, heads, query_length, query_dim]
     * 1: key    [batch, heads, key_length, query_dim]
     * 2: value  [batch, heads, key_length, value_dim]
     * 3: dropout_p
     * 4: is_causal
     * 5: return_debug_mask
     * 6: scale oppure None
     */
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

    const at::Tensor query =
        inputs[0].toTensor();

    const at::Tensor key =
        inputs[1].toTensor();

    const at::Tensor value =
        inputs[2].toTensor();

    if (
        !query.defined() ||
        !key.defined() ||
        !value.defined()
    ) {
        return 0;
    }

    if (
        query.dim() != 4 ||
        key.dim() != 4 ||
        value.dim() != 4
    ) {
        return 0;
    }

    const std::uint64_t batch =
        dimension_to_uint64(query.size(0));

    const std::uint64_t heads =
        dimension_to_uint64(query.size(1));

    const std::uint64_t query_length =
        dimension_to_uint64(query.size(2));

    const std::uint64_t key_length =
        dimension_to_uint64(key.size(2));

    const std::uint64_t query_dimension =
        dimension_to_uint64(query.size(3));

    const std::uint64_t value_dimension =
        dimension_to_uint64(value.size(3));

    /*
     * Verifica compatibilità delle forme.
     */
    if (
        dimension_to_uint64(key.size(0)) != batch ||
        dimension_to_uint64(value.size(0)) != batch ||
        dimension_to_uint64(key.size(1)) != heads ||
        dimension_to_uint64(value.size(1)) != heads ||
        dimension_to_uint64(key.size(3)) != query_dimension ||
        dimension_to_uint64(value.size(2)) != key_length
    ) {
        return 0;
    }

    /*
     * Q x K^T:
     * 2 * B * H * Lq * Lk * Dq
     */
    std::uint64_t attention_scores =
        checked_multiply(batch, heads);

    attention_scores =
        checked_multiply(
            attention_scores,
            query_length
        );

    attention_scores =
        checked_multiply(
            attention_scores,
            key_length
        );

    const std::uint64_t qk_flops =
        checked_multiply(
            checked_multiply(
                attention_scores,
                query_dimension
            ),
            2
        );

    /*
     * Softmax:
     * 3 * B * H * Lq * Lk
     *
     * Mantiene la convenzione della precedente
     * implementazione FLOPpy.
     */
    const std::uint64_t softmax_flops =
        checked_multiply(
            attention_scores,
            3
        );

    /*
     * Attention probabilities x V:
     * 2 * B * H * Lq * Lk * Dv
     */
    const std::uint64_t value_flops =
        checked_multiply(
            checked_multiply(
                attention_scores,
                value_dimension
            ),
            2
        );

    return checked_add(
        checked_add(
            qk_flops,
            softmax_flops
        ),
        value_flops
    );
}

}  // namespace floppy::native
