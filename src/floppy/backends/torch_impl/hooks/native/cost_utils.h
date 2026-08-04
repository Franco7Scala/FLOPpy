#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace floppy::native {

inline std::uint64_t checked_multiply(
    const std::uint64_t first,
    const std::uint64_t second
) {
    if (
        first != 0 &&
        second >
            std::numeric_limits<std::uint64_t>::max() / first
    ) {
        throw std::overflow_error(
            "FLOP/BOP count overflow during multiplication."
        );
    }

    return first * second;
}

inline std::uint64_t checked_add(
    const std::uint64_t first,
    const std::uint64_t second
) {
    if (
        second >
        std::numeric_limits<std::uint64_t>::max() - first
    ) {
        throw std::overflow_error(
            "FLOP/BOP count overflow during addition."
        );
    }

    return first + second;
}

inline std::uint64_t dimension_to_uint64(
    const int64_t dimension
) {
    if (dimension < 0) {
        throw std::runtime_error(
            "Negative tensor dimension encountered."
        );
    }

    return static_cast<std::uint64_t>(dimension);
}

inline std::uint64_t tensor_numel(
    const at::Tensor& tensor
) {
    if (!tensor.defined()) {
        return 0;
    }

    const int64_t numel = tensor.numel();

    if (numel < 0) {
        throw std::runtime_error(
            "Invalid tensor element count."
        );
    }

    return static_cast<std::uint64_t>(numel);
}

}  // namespace floppy::native
