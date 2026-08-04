#include "elementwise_ops.h"

#include "../cost_utils.h"

namespace floppy::native {

/**
 * Calcola i FLOP di aten::add_.
 *
 * Operazione element-wise:
 *
 * self += alpha * other
 *
 * Per compatibilità con la precedente implementazione
 * di FLOPpy viene conteggiato un FLOP per elemento
 * del tensore aggiornato.
 */
std::uint64_t calculate_add_inplace_flops(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    if (
        inputs.empty() ||
        !inputs[0].isTensor()
    ) {
        return 0;
    }

    const at::Tensor self =
        inputs[0].toTensor();

    if (!self.defined()) {
        return 0;
    }

    return tensor_numel(self);
}

}  // namespace floppy::native
