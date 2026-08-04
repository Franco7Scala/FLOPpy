#pragma once

#include <ATen/record_function.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace floppy::native {

using FlopCalculator = std::uint64_t (*)(
    const at::RecordFunction&
);

class OperatorRegistry {
public:
    using RegistryMap =
        std::unordered_map<
            std::string,
            FlopCalculator
        >;

    static const OperatorRegistry& instance();

    FlopCalculator find(
        const std::string& operator_name
    ) const;

    bool contains(
        const std::string& operator_name
    ) const;

    std::vector<std::string>
    supported_operators() const;

private:
    OperatorRegistry();

    OperatorRegistry(
        const OperatorRegistry&
    ) = delete;

    OperatorRegistry& operator=(
        const OperatorRegistry&
    ) = delete;

    RegistryMap calculators_;
};

}  // namespace floppy::native
