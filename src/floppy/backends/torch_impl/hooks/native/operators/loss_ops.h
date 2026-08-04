#pragma once

    #include <ATen/record_function.h>

    #include <cstdint>

    namespace floppy::native {

    std::uint64_t calculate_mse_loss_flops(
    const at::RecordFunction& function
);

    }  // namespace floppy::native
