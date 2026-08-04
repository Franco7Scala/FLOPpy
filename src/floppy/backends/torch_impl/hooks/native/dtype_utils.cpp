#include "dtype_utils.h"

#include "cost_utils.h"

namespace floppy::native {

std::uint64_t tensor_bit_width(
    const at::Tensor& tensor
) {
    if (!tensor.defined()) {
        return 0;
    }

    const std::size_t bytes_per_element =
        tensor.element_size();

    if (bytes_per_element == 0) {
        return 0;
    }

    return checked_multiply(
        static_cast<std::uint64_t>(
            bytes_per_element
        ),
        8
    );
}

std::uint64_t get_effective_bit_width(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    for (const c10::IValue& value : inputs) {
        if (value.isTensor()) {
            const std::uint64_t bit_width =
                tensor_bit_width(
                    value.toTensor()
                );

            if (bit_width != 0) {
                return bit_width;
            }
        }

        if (value.isTensorList()) {
            const auto tensors =
                value.toTensorVector();

            for (const at::Tensor& tensor : tensors) {
                const std::uint64_t bit_width =
                    tensor_bit_width(tensor);

                if (bit_width != 0) {
                    return bit_width;
                }
            }
        }
    }

    return 32;
}

}  // namespace floppy::native
