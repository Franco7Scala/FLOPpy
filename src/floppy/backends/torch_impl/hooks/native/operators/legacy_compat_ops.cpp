#include "legacy_compat_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

namespace floppy::native {
namespace {

std::uint64_t first_tensor_numel(const at::RecordFunction& function) {
    const auto inputs = function.inputs();
    if (inputs.empty() || !inputs[0].isTensor()) return 0;

    const at::Tensor tensor = inputs[0].toTensor();
    return tensor.defined() ? tensor_numel(tensor) : 0;
}

std::uint64_t first_tensor_list_numel(const at::RecordFunction& function) {
    const auto inputs = function.inputs();
    if (inputs.empty() || !inputs[0].isTensorList()) return 0;

    std::uint64_t total = 0;
    for (const at::Tensor& tensor : inputs[0].toTensorVector()) {
        if (!tensor.defined()) return 0;
        total = checked_add(total, tensor_numel(tensor));
    }

    return total;
}

}  // namespace

std::uint64_t calculate_inplace_elementwise_flops(const at::RecordFunction& function) {
    return first_tensor_numel(function);
}

std::uint64_t calculate_complex_elementwise_flops(const at::RecordFunction& function) {
    return checked_multiply(first_tensor_numel(function), 3);
}

std::uint64_t calculate_foreach_elementwise_flops(const at::RecordFunction& function) {
    return first_tensor_list_numel(function);
}

std::uint64_t calculate_foreach_complex_flops(const at::RecordFunction& function) {
    return checked_multiply(first_tensor_list_numel(function), 3);
}

std::uint64_t calculate_baddbmm_flops(const at::RecordFunction& function) {
    const auto inputs = function.inputs();
    if (inputs.size() < 3 || !inputs[1].isTensor() || !inputs[2].isTensor()) return 0;

    const at::Tensor batch1 = inputs[1].toTensor();
    const at::Tensor batch2 = inputs[2].toTensor();

    if (!batch1.defined() || !batch2.defined() || batch1.dim() != 3 || batch2.dim() != 3) return 0;
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) return 0;

    std::uint64_t flops = 2;
    flops = checked_multiply(flops, dimension_to_uint64(batch1.size(0)));
    flops = checked_multiply(flops, dimension_to_uint64(batch1.size(1)));
    flops = checked_multiply(flops, dimension_to_uint64(batch2.size(2)));
    flops = checked_multiply(flops, dimension_to_uint64(batch1.size(2)));
    return flops;
}

std::uint64_t calculate_index_select_flops(const at::RecordFunction& function) {
    const auto inputs = function.inputs();
    if (inputs.size() < 3 || !inputs[0].isTensor() || !inputs[1].isInt() || !inputs[2].isTensor()) return 0;

    const at::Tensor input = inputs[0].toTensor();
    const at::Tensor index = inputs[2].toTensor();
    if (!input.defined() || !index.defined() || input.dim() <= 0) return 0;

    int64_t dim = inputs[1].toInt();
    if (dim < 0) dim += input.dim();
    if (dim < 0 || dim >= input.dim()) return 0;

    std::uint64_t output_elements = tensor_numel(index);
    for (int64_t current = 0; current < input.dim(); ++current) {
        if (current == dim) continue;
        output_elements = checked_multiply(
            output_elements,
            dimension_to_uint64(input.size(current))
        );
    }

    return output_elements;
}

std::uint64_t calculate_rms_norm_flops(const at::RecordFunction& function) {
    return checked_multiply(first_tensor_numel(function), 4);
}

}  // namespace floppy::native
