#include "recurrent_ops.h"

#include "../cost_utils.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace floppy::native {

namespace {

enum class RecurrentKind {
    Rnn,
    Gru,
    Lstm,
    Unsupported
};

RecurrentKind get_recurrent_kind(
    const std::string& operator_name
) {
    if (
        operator_name == "aten::rnn_tanh" ||
        operator_name == "aten::rnn_relu"
    ) {
        return RecurrentKind::Rnn;
    }

    if (operator_name == "aten::gru") {
        return RecurrentKind::Gru;
    }

    if (operator_name == "aten::lstm") {
        return RecurrentKind::Lstm;
    }

    return RecurrentKind::Unsupported;
}

std::uint64_t recurrent_elementwise_factor(
    const RecurrentKind kind
) {
    switch (kind) {
        case RecurrentKind::Rnn:
            return 2;
        case RecurrentKind::Gru:
            return 7;
        case RecurrentKind::Lstm:
            return 10;
        case RecurrentKind::Unsupported:
            return 0;
    }

    return 0;
}

}  // namespace

std::uint64_t calculate_recurrent_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    const char* raw_name = function.name();
    const std::string operator_name =
        raw_name != nullptr ? std::string(raw_name) : std::string();

    const RecurrentKind kind =
        get_recurrent_kind(operator_name);

    if (kind == RecurrentKind::Unsupported) {
        return 0;
    }

    if (
        inputs.size() < 9 ||
        !inputs[0].isTensor() ||
        !inputs[2].isTensorList() ||
        !inputs[3].isBool() ||
        !inputs[4].isInt() ||
        !inputs[7].isBool() ||
        !inputs[8].isBool()
    ) {
        return 0;
    }

    const at::Tensor input = inputs[0].toTensor();

    if (!input.defined() || input.dim() != 3) {
        return 0;
    }

    const bool has_biases = inputs[3].toBool();
    const int64_t num_layers_signed = inputs[4].toInt();
    const bool bidirectional = inputs[7].toBool();
    const bool batch_first = inputs[8].toBool();

    if (num_layers_signed <= 0) {
        return 0;
    }

    const std::uint64_t num_layers =
        dimension_to_uint64(num_layers_signed);
    const std::uint64_t directions =
        bidirectional ? 2 : 1;
    const std::uint64_t recurrent_blocks =
        checked_multiply(num_layers, directions);

    const std::uint64_t batch =
        dimension_to_uint64(input.size(batch_first ? 0 : 1));
    const std::uint64_t sequence_length =
        dimension_to_uint64(input.size(batch_first ? 1 : 0));
    const std::uint64_t batch_timesteps =
        checked_multiply(batch, sequence_length);

    const std::vector<at::Tensor> parameters =
        inputs[2].toTensorVector();

    const std::size_t tensors_per_block =
        has_biases ? 4 : 2;
    const std::uint64_t expected_parameters =
        checked_multiply(
            recurrent_blocks,
            static_cast<std::uint64_t>(tensors_per_block)
        );

    if (
        parameters.size() !=
        static_cast<std::size_t>(expected_parameters)
    ) {
        return 0;
    }

    const std::uint64_t elementwise_factor =
        recurrent_elementwise_factor(kind);

    std::uint64_t total_flops = 0;

    for (
        std::uint64_t block = 0;
        block < recurrent_blocks;
        ++block
    ) {
        const std::size_t offset =
            static_cast<std::size_t>(block) * tensors_per_block;

        const at::Tensor& weight_ih = parameters[offset];
        const at::Tensor& weight_hh = parameters[offset + 1];

        if (
            !weight_ih.defined() ||
            !weight_hh.defined() ||
            weight_ih.dim() != 2 ||
            weight_hh.dim() != 2
        ) {
            return 0;
        }

        const std::uint64_t weight_elements =
            checked_add(
                tensor_numel(weight_ih),
                tensor_numel(weight_hh)
            );

        std::uint64_t block_flops =
            checked_multiply(
                checked_multiply(
                    batch_timesteps,
                    weight_elements
                ),
                2
            );

        if (has_biases) {
            const at::Tensor& bias_ih = parameters[offset + 2];
            const at::Tensor& bias_hh = parameters[offset + 3];

            if (!bias_ih.defined() || !bias_hh.defined()) {
                return 0;
            }

            const std::uint64_t bias_elements =
                checked_add(
                    tensor_numel(bias_ih),
                    tensor_numel(bias_hh)
                );

            block_flops = checked_add(
                block_flops,
                checked_multiply(
                    batch_timesteps,
                    bias_elements
                )
            );
        }

        const std::uint64_t hidden_size =
            dimension_to_uint64(weight_hh.size(1));

        const std::uint64_t elementwise_flops =
            checked_multiply(
                checked_multiply(
                    batch_timesteps,
                    hidden_size
                ),
                elementwise_factor
            );

        block_flops =
            checked_add(block_flops, elementwise_flops);
        total_flops =
            checked_add(total_flops, block_flops);
    }

    return total_flops;
}

}  // namespace floppy::native
