#include "operator_registry.h"

#include "operators/attention_ops.h"
#include "operators/convolution_ops.h"
#include "operators/elementwise_ops.h"
#include "operators/embedding_ops.h"
#include "operators/indexing_ops.h"
#include "operators/linear_ops.h"
#include "operators/loss_ops.h"
#include "operators/normalization_ops.h"
#include "operators/optimizer_ops.h"
#include "operators/pooling_ops.h"
#include "operators/recurrent_ops.h"
#include "operators/sorting_ops.h"
#include "operators/upsampling_ops.h"

#include <algorithm>

namespace floppy::native {

const OperatorRegistry& OperatorRegistry::instance() {
    static const OperatorRegistry registry;
    return registry;
}

FlopCalculator OperatorRegistry::find(
    const std::string& operator_name
) const {
    const auto iterator = calculators_.find(operator_name);
    return iterator == calculators_.end() ? nullptr : iterator->second;
}

bool OperatorRegistry::contains(
    const std::string& operator_name
) const {
    return calculators_.find(operator_name) != calculators_.end();
}

std::vector<std::string> OperatorRegistry::supported_operators() const {
    std::vector<std::string> names;
    names.reserve(calculators_.size());
    for (const auto& entry : calculators_) { names.push_back(entry.first); }
    std::sort(names.begin(), names.end());
    return names;
}

OperatorRegistry::OperatorRegistry()
    : calculators_{
        {"aten::add", &calculate_binary_elementwise_flops},
        {"aten::add_", &calculate_add_inplace_flops},
        {"aten::sub", &calculate_binary_elementwise_flops},
        {"aten::mul", &calculate_binary_elementwise_flops},
        {"aten::div", &calculate_binary_elementwise_flops},
        {"aten::exp", &calculate_unary_elementwise_flops},
        {"aten::log", &calculate_unary_elementwise_flops},
        {"aten::pow", &calculate_binary_elementwise_flops},
        {"aten::neg", &calculate_unary_elementwise_flops},
        {"aten::abs", &calculate_abs_flops},
        {"aten::relu", &calculate_unary_elementwise_flops},
        {"aten::sigmoid", &calculate_unary_elementwise_flops},
        {"aten::tanh", &calculate_unary_elementwise_flops},
        {"aten::sqrt", &calculate_unary_elementwise_flops},
        {"aten::rsqrt", &calculate_unary_elementwise_flops},
        {"aten::gelu", &calculate_unary_elementwise_flops},
        {"aten::silu", &calculate_unary_elementwise_flops},
        {"aten::mish", &calculate_unary_elementwise_flops},
        {"aten::sub_", &calculate_inplace_elementwise_flops},
        {"aten::exp_", &calculate_inplace_elementwise_flops},
        {"aten::log_", &calculate_inplace_elementwise_flops},
        {"aten::pow_", &calculate_inplace_elementwise_flops},
        {"aten::neg_", &calculate_inplace_elementwise_flops},
        {"aten::abs_", &calculate_inplace_elementwise_flops},
        {"aten::relu_", &calculate_inplace_elementwise_flops},
        {"aten::sigmoid_", &calculate_inplace_elementwise_flops},
        {"aten::tanh_", &calculate_inplace_elementwise_flops},
        {"aten::sqrt_", &calculate_inplace_elementwise_flops},
        {"aten::rsqrt_", &calculate_inplace_elementwise_flops},
        {"aten::gelu_", &calculate_inplace_elementwise_flops},
        {"aten::silu_", &calculate_inplace_elementwise_flops},
        {"aten::mish_", &calculate_inplace_elementwise_flops},
        {"aten::addcmul", &calculate_complex_elementwise_flops},
        {"aten::addcdiv", &calculate_complex_elementwise_flops},
        {"aten::_foreach_add", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_add_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_sub", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_sub_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_mul", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_mul_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_div", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_div_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_sqrt", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_sqrt_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_exp", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_exp_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_neg", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_neg_", &calculate_foreach_elementwise_flops},
        {"aten::_foreach_addcmul", &calculate_foreach_complex_flops},
        {"aten::_foreach_addcmul_", &calculate_foreach_complex_flops},
        {"aten::_foreach_addcdiv", &calculate_foreach_complex_flops},
        {"aten::_foreach_addcdiv_", &calculate_foreach_complex_flops},
        {"aten::mul_", &calculate_optimizer_unary_inplace_flops},
        {"aten::div_", &calculate_optimizer_unary_inplace_flops},
        {"aten::lerp_", &calculate_lerp_inplace_flops},
        {"aten::addcmul_", &calculate_complex_optimizer_inplace_flops},
        {"aten::addcdiv_", &calculate_complex_optimizer_inplace_flops},
        {"aten::native_layer_norm", &calculate_native_layer_norm_flops},
        {"aten::rms_norm", &calculate_rms_norm_flops},
        {"aten::_softmax", &calculate_softmax_flops},
        {"aten::_log_softmax", &calculate_softmax_flops},
        {"aten::mse_loss", &calculate_mse_loss_flops},
        {"aten::nll_loss_forward", &calculate_nll_loss_forward_flops},
        {"aten::embedding", &calculate_embedding_flops},
        {"aten::_scaled_dot_product_flash_attention", &calculate_flash_attention_flops},
        {"aten::scatter", &calculate_destination_indexing_flops},
        {"aten::scatter_add", &calculate_destination_indexing_flops},
        {"aten::scatter_reduce", &calculate_destination_indexing_flops},
        {"aten::index_add", &calculate_destination_indexing_flops},
        {"aten::index_select", &calculate_index_select_flops},
        {"aten::gather", &calculate_gather_flops},
        {"aten::bincount", &calculate_bincount_flops},
        {"aten::max_pool1d", &calculate_pooling_flops},
        {"aten::max_pool2d_with_indices", &calculate_pooling_flops},
        {"aten::max_pool3d_with_indices", &calculate_pooling_flops},
        {"aten::avg_pool2d", &calculate_pooling_flops},
        {"aten::avg_pool3d", &calculate_pooling_flops},
        {"aten::adaptive_avg_pool1d", &calculate_adaptive_avg_pooling_flops},
        {"aten::adaptive_avg_pool2d", &calculate_adaptive_avg_pooling_flops},
        {"aten::adaptive_avg_pool3d", &calculate_adaptive_avg_pooling_flops},
        {"aten::upsample_nearest1d", &calculate_upsampling_flops},
        {"aten::_upsample_nearest_exact1d", &calculate_upsampling_flops},
        {"aten::upsample_linear1d", &calculate_upsampling_flops},
        {"aten::upsample_nearest2d", &calculate_upsampling_flops},
        {"aten::_upsample_nearest_exact2d", &calculate_upsampling_flops},
        {"aten::upsample_bilinear2d", &calculate_upsampling_flops},
        {"aten::_upsample_bilinear2d_aa", &calculate_upsampling_flops},
        {"aten::upsample_bicubic2d", &calculate_upsampling_flops},
        {"aten::upsample_nearest3d", &calculate_upsampling_flops},
        {"aten::_upsample_nearest_exact3d", &calculate_upsampling_flops},
        {"aten::upsample_trilinear3d", &calculate_upsampling_flops},
        {"aten::topk", &calculate_sorting_flops},
        {"aten::sort", &calculate_sorting_flops},
        {"aten::argsort", &calculate_sorting_flops},
        {"aten::rnn_tanh", &calculate_recurrent_flops},
        {"aten::rnn_relu", &calculate_recurrent_flops},
        {"aten::gru", &calculate_recurrent_flops},
        {"aten::lstm", &calculate_recurrent_flops},
        {"aten::dot", &calculate_dot_flops},
        {"aten::mv", &calculate_mv_flops},
        {"aten::mm", &calculate_mm_flops},
        {"aten::bmm", &calculate_bmm_flops},
        {"aten::baddbmm", &calculate_baddbmm_flops},
        {"aten::addmm", &calculate_addmm_flops},
        {"aten::_convolution", &calculate_convolution_flops}
    } {}

}  // namespace floppy::native
