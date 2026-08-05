#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Calcola i FLOP di aten::_scaled_dot_product_flash_attention.
 *
 * Il wrapper aten::scaled_dot_product_attention non viene
 * conteggiato, così da evitare duplicazioni.
 *
 * I backend matematici decomposti continuano invece a essere
 * conteggiati tramite bmm, softmax e operatori element-wise.
 */
std::uint64_t calculate_flash_attention_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
