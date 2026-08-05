#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Calcola il costo storico FLOPpy di aten::embedding.
 *
 * Convenzione legacy:
 *
 * FLOP = numero_indici * dimensione_embedding
 *
 * Nota: si tratta di una metrica di costo mantenuta per
 * compatibilità con la precedente implementazione, anche se
 * l'embedding è principalmente un'operazione di lookup/memoria.
 */
std::uint64_t calculate_embedding_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
