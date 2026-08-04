#include "loss_ops.h"

#include "../cost_utils.h"

namespace floppy::native {

/**
 * Calcola i FLOP di aten::mse_loss.
 *
 * Firma:
 *
 * 0: input
 * 1: target
 * 2: reduction
 *
 * reduction:
 *   0 = none
 *   1 = mean
 *   2 = sum
 *
 * Convenzione:
 *
 * Per ogni elemento:
 *   - una sottrazione;
 *   - una moltiplicazione per il quadrato.
 *
 * Riduzione sum:
 *   - circa una addizione per elemento.
 *
 * Riduzione mean:
 *   - stessa riduzione di sum;
 *   - una divisione finale.
 */
std::uint64_t calculate_mse_loss_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (inputs.size() < 3) {
        return 0;
    }

    if (
        !inputs[0].isTensor() ||
        !inputs[1].isTensor() ||
        !inputs[2].isInt()
    ) {
        return 0;
    }

    const at::Tensor input =
        inputs[0].toTensor();

    const at::Tensor target =
        inputs[1].toTensor();

    if (
        !input.defined() ||
        !target.defined()
    ) {
        return 0;
    }

    if (input.sizes() != target.sizes()) {
        return 0;
    }

    const std::uint64_t elements =
        tensor_numel(input);

    const int64_t reduction =
        inputs[2].toInt();

    /*
     * Sottrazione + quadrato:
     * 2 FLOP per elemento.
     */
    std::uint64_t flops =
        checked_multiply(
            elements,
            2
        );

    /*
     * reduction == 0:
     * nessuna riduzione.
     */
    if (reduction == 0) {
        return flops;
    }

    /*
     * reduction == 1 o 2:
     * somma degli elementi.
     */
    if (
        reduction == 1 ||
        reduction == 2
    ) {
        flops = checked_add(
            flops,
            elements
        );
    }
    else {
        return 0;
    }

    /*
     * reduction == 1:
     * divisione finale per la media.
     */
    if (reduction == 1) {
        flops = checked_add(
            flops,
            1
        );
    }

    return flops;
}

}  // namespace floppy::native
