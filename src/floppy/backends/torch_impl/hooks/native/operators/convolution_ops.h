#pragma once

#include <ATen/record_function.h>

#include <cstdint>

namespace floppy::native {

/**
 * Calcola i FLOP dell'operatore canonico aten::_convolution.
 *
 * Supporta convoluzioni standard e trasposte 1D, 2D e 3D,
 * incluse le varianti grouped e depthwise.
 */
std::uint64_t calculate_convolution_flops(
    const at::RecordFunction& function
);

}  // namespace floppy::native
