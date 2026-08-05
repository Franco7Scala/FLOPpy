#include "diagnostics.h"

#include <ATen/ATen.h>

#include <sstream>
#include <string>
#include <vector>

namespace floppy::native {

/**
 * Returns a human-readable description of the shape of a tensor.
 */
static std::string describe_tensor(
    const at::Tensor& tensor
) {
    if (!tensor.defined()) {
        return "Tensor(undefined)";
    }

    std::ostringstream stream;

    stream << "Tensor(shape=(";

    const auto sizes = tensor.sizes();

    for (
        std::size_t dimension = 0;
        dimension < static_cast<std::size_t>(sizes.size());
        ++dimension
    ) {
        if (dimension > 0) {
            stream << ", ";
        }

        stream << sizes[dimension];
    }

    if (sizes.size() == 1) {
        stream << ",";
    }

    stream << "))";

    return stream.str();
}

/**
 * Returns a human-readable description of a list of integers.
 */
static std::string describe_int_list(
    const c10::IValue& value
) {
    const std::vector<int64_t> values =
        value.toIntVector();

    std::ostringstream stream;

    stream << "int_list([";

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (index > 0) {
            stream << ", ";
        }

        stream << values[index];
    }

    stream << "])";

    return stream.str();
}

/**
 * Returns a human-readable description of an IValue.
 */
static std::string describe_value(
    const c10::IValue& value
) {
    if (value.isTensor()) {
        return describe_tensor(value.toTensor());
    }

    if (value.isInt()) {
        return (
            "int(" +
            std::to_string(value.toInt()) +
            ")"
        );
    }

    if (value.isDouble()) {
        std::ostringstream stream;

        stream << "float("
               << value.toDouble()
               << ")";

        return stream.str();
    }

    if (value.isBool()) {
        return value.toBool()
            ? "bool(true)"
            : "bool(false)";
    }

    if (value.isNone()) {
        return "None";
    }

    if (value.isIntList()) {
        return describe_int_list(value);
    }

    if (value.isString()) {
        return (
            "str(\"" +
            value.toStringRef() +
            "\")"
        );
    }

    if (value.isDevice()) {
        std::ostringstream stream;

        stream << "device("
               << value.toDevice()
               << ")";

        return stream.str();
    }

    if (value.isTensorList()) {
        const auto tensors =
            value.toTensorVector();

        std::ostringstream stream;

        stream << "tensor_list([";

        for (
            std::size_t index = 0;
            index < tensors.size();
            ++index
        ) {
            if (index > 0) {
                stream << ", ";
            }

            stream << describe_tensor(
                tensors[index]
            );
        }

        stream << "])";

        return stream.str();
    }

    return "other";
}

/**
 * Returns a human-readable description of all inputs of the intercepted operator.
 */
std::string describe_inputs(
    const at::RecordFunction& function
) {
    const auto inputs = function.inputs();

    std::ostringstream stream;

    stream << "[";

    for (
        std::size_t index = 0;
        index < static_cast<std::size_t>(inputs.size());
        ++index
    ) {
        if (index > 0) {
            stream << ", ";
        }

        stream << describe_value(inputs[index]);
    }

    stream << "]";

    return stream.str();
}

}  // namespace floppy::native
