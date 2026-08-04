#include "operator_registry.h"

#include "operators/convolution_ops.h"
#include "operators/elementwise_ops.h"
#include "operators/linear_ops.h"
#include "operators/loss_ops.h"

#include <algorithm>

namespace floppy::native {

const OperatorRegistry& OperatorRegistry::instance() {
    static const OperatorRegistry registry;
    return registry;
}

FlopCalculator OperatorRegistry::find(
    const std::string& operator_name
) const {
    const auto iterator =
        calculators_.find(operator_name);

    if (iterator == calculators_.end()) {
        return nullptr;
    }

    return iterator->second;
}

bool OperatorRegistry::contains(
    const std::string& operator_name
) const {
    return (
        calculators_.find(operator_name) !=
        calculators_.end()
    );
}

std::vector<std::string>
OperatorRegistry::supported_operators() const {
    std::vector<std::string> names;
    names.reserve(calculators_.size());

    for (const auto& entry : calculators_) {
        names.push_back(entry.first);
    }

    std::sort(names.begin(), names.end());
    return names;
}

OperatorRegistry::OperatorRegistry()
    : calculators_{
        {
            "aten::add_",
            &calculate_add_inplace_flops
        },
        {
            "aten::mse_loss",
            &calculate_mse_loss_flops
        },
        {
            "aten::dot",
            &calculate_dot_flops
        },
        {
            "aten::mv",
            &calculate_mv_flops
        },
        {
            "aten::mm",
            &calculate_mm_flops
        },
        {
            "aten::bmm",
            &calculate_bmm_flops
        },
        {
            "aten::addmm",
            &calculate_addmm_flops
        },
        {
            "aten::mkldnn_convolution",
            &calculate_mkldnn_convolution_flops
        }
    } {
}

}  // namespace floppy::native
