#include "native_counter.h"

#include "cost_utils.h"
#include "diagnostics.h"
#include "dtype_utils.h"
#include "operator_registry.h"

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace floppy::native {

class NativeCounterProbe;

thread_local NativeCounterProbe* tls_active_probe = nullptr;

class NativeCounterObserverContext final
    : public at::ObserverContext {
public:
    NativeCounterObserverContext(
        NativeCounterProbe* probe,
        bool entered_canonical_scope
    )
        : probe_(probe),
          entered_canonical_scope_(entered_canonical_scope) {
    }

    NativeCounterProbe* probe() const {
        return probe_;
    }

    bool entered_canonical_scope() const {
        return entered_canonical_scope_;
    }

private:
    NativeCounterProbe* probe_ = nullptr;
    bool entered_canonical_scope_ = false;
};

class NativeCounterProbe {
public:
    NativeCounterProbe() = default;

    NativeCounterProbe(const NativeCounterProbe&) = delete;
    NativeCounterProbe& operator=(const NativeCounterProbe&) = delete;
    NativeCounterProbe(NativeCounterProbe&&) = delete;
    NativeCounterProbe& operator=(NativeCounterProbe&&) = delete;

    ~NativeCounterProbe() {
        stop_noexcept();
    }

    void start() {
        if (active_) {
            throw std::runtime_error(
                "NativeCounterProbe is already active."
            );
        }

        if (tls_active_probe != nullptr) {
            throw std::runtime_error(
                "Another NativeCounterProbe is already active "
                "in the current thread."
            );
        }

        paused_.store(false, std::memory_order_relaxed);
        canonical_scope_stack_.clear();

        tls_active_probe = this;
        active_ = true;

        try {
            at::RecordFunctionCallback callback(
                &NativeCounterProbe::on_function_start,
                &NativeCounterProbe::on_function_end
            );

            callback
                .needsInputs(true)
                .needsOutputs(false)
                .needsIds(false)
                .samplingProb(1.0)
                .scopes(
                    std::unordered_set<at::RecordScope>{
                        at::RecordScope::FUNCTION
                    }
                );

            callback_handle_ =
                at::addThreadLocalCallback(
                    std::move(callback)
                );
        }
        catch (...) {
            active_ = false;
            canonical_scope_stack_.clear();
            tls_active_probe = nullptr;
            callback_handle_ = at::INVALID_CALLBACK_HANDLE;
            throw;
        }
    }

    void stop() {
        if (!active_) {
            return;
        }

        if (callback_handle_ != at::INVALID_CALLBACK_HANDLE) {
            at::removeCallback(callback_handle_);
            callback_handle_ = at::INVALID_CALLBACK_HANDLE;
        }

        paused_.store(false, std::memory_order_relaxed);
        canonical_scope_stack_.clear();
        active_ = false;

        if (tls_active_probe == this) {
            tls_active_probe = nullptr;
        }
    }

    void reset() {
        event_count_.store(0, std::memory_order_relaxed);
        total_flops_.store(0, std::memory_order_relaxed);
        total_bops_.store(0, std::memory_order_relaxed);
        paused_.store(false, std::memory_order_relaxed);

        canonical_scope_stack_.clear();
        operator_counts_.clear();
        operator_input_signatures_.clear();
        operator_flops_.clear();
        operator_bops_.clear();
        operator_errors_.clear();
    }

    void pause() {
        paused_.store(true, std::memory_order_relaxed);
    }

    void resume() {
        paused_.store(false, std::memory_order_relaxed);
    }

    void add_flops(std::uint64_t value) {
        const std::uint64_t current =
            total_flops_.load(std::memory_order_relaxed);

        total_flops_.store(
            checked_add(current, value),
            std::memory_order_relaxed
        );
    }

    void add_bops(std::uint64_t value) {
        const std::uint64_t current =
            total_bops_.load(std::memory_order_relaxed);

        total_bops_.store(
            checked_add(current, value),
            std::memory_order_relaxed
        );
    }

    bool active() const {
        return active_;
    }

    bool paused() const {
        return paused_.load(std::memory_order_relaxed);
    }

    std::uint64_t event_count() const {
        return event_count_.load(std::memory_order_relaxed);
    }

    std::uint64_t total_flops() const {
        return total_flops_.load(std::memory_order_relaxed);
    }

    std::uint64_t total_bops() const {
        return total_bops_.load(std::memory_order_relaxed);
    }

    std::unordered_map<std::string, std::uint64_t>
    operator_counts() const {
        return operator_counts_;
    }

    std::unordered_map<std::string, std::string>
    operator_input_signatures() const {
        return operator_input_signatures_;
    }

    std::unordered_map<std::string, std::uint64_t>
    operator_flops() const {
        return operator_flops_;
    }

    std::unordered_map<std::string, std::uint64_t>
    operator_bops() const {
        return operator_bops_;
    }

    std::unordered_map<std::string, std::string>
    operator_errors() const {
        return operator_errors_;
    }

    std::vector<std::string>
    supported_operators() const {
        return OperatorRegistry::instance()
            .supported_operators();
    }

private:
    struct CanonicalScopeState {
        bool has_contribution = false;
    };

    static bool opens_canonical_scope(
        const std::string& operator_name
    ) {
        static const std::unordered_set<std::string>
            canonical_scope_operators = {
                "aten::adaptive_avg_pool1d",
                "aten::adaptive_avg_pool2d",
                "aten::adaptive_avg_pool3d",

                "aten::upsample_nearest1d",
                "aten::_upsample_nearest_exact1d",
                "aten::upsample_linear1d",
                "aten::upsample_nearest2d",
                "aten::_upsample_nearest_exact2d",
                "aten::upsample_bilinear2d",
                "aten::_upsample_bilinear2d_aa",
                "aten::upsample_bicubic2d",
                "aten::upsample_nearest3d",
                "aten::_upsample_nearest_exact3d",
                "aten::upsample_trilinear3d",

                "aten::sort",
                "aten::argsort",

                "aten::embedding",

                "aten::sub_",
                "aten::exp_",
                "aten::log_",
                "aten::pow_",
                "aten::neg_",
                "aten::abs_",
                "aten::relu_",
                "aten::sigmoid_",
                "aten::tanh_",
                "aten::sqrt_",
                "aten::rsqrt_",
                "aten::gelu_",
                "aten::silu_",
                "aten::mish_",

                "aten::_foreach_add",
                "aten::_foreach_add_",
                "aten::_foreach_sub",
                "aten::_foreach_sub_",
                "aten::_foreach_mul",
                "aten::_foreach_mul_",
                "aten::_foreach_div",
                "aten::_foreach_div_",
                "aten::_foreach_sqrt",
                "aten::_foreach_sqrt_",
                "aten::_foreach_exp",
                "aten::_foreach_exp_",
                "aten::_foreach_neg",
                "aten::_foreach_neg_",
                "aten::_foreach_addcmul",
                "aten::_foreach_addcmul_",
                "aten::_foreach_addcdiv",
                "aten::_foreach_addcdiv_",

                "aten::rnn_tanh",
                "aten::rnn_relu",
                "aten::gru",
                "aten::lstm"
            };

        return canonical_scope_operators.find(operator_name) !=
               canonical_scope_operators.end();
    }

    bool canonical_ancestor_has_contribution() const {
        for (const CanonicalScopeState& state : canonical_scope_stack_) {
            if (state.has_contribution) {
                return true;
            }
        }

        return false;
    }

    void mark_canonical_scopes_contributed() {
        for (CanonicalScopeState& state : canonical_scope_stack_) {
            state.has_contribution = true;
        }
    }

    static std::unique_ptr<at::ObserverContext>
    on_function_start(
        const at::RecordFunction& function
    ) {
        NativeCounterProbe* probe = tls_active_probe;

        if (
            probe == nullptr ||
            !probe->active_ ||
            probe->paused()
        ) {
            return nullptr;
        }

        const char* raw_name = function.name();

        const std::string operator_name =
            raw_name != nullptr
                ? std::string(raw_name)
                : std::string("<unnamed>");

        const bool enters_canonical_scope =
            opens_canonical_scope(operator_name);

        bool suppress_flop_calculation = false;

        if (!probe->canonical_scope_stack_.empty()) {
            suppress_flop_calculation =
                !enters_canonical_scope ||
                probe->canonical_ancestor_has_contribution();
        }

        const bool contributed =
            probe->record_function(
                function,
                suppress_flop_calculation
            );

        if (contributed) {
            probe->mark_canonical_scopes_contributed();
        }

        if (enters_canonical_scope) {
            probe->canonical_scope_stack_.push_back(
                CanonicalScopeState{contributed}
            );
        }

        return std::make_unique<NativeCounterObserverContext>(
            probe,
            enters_canonical_scope
        );
    }

    static void on_function_end(
        const at::RecordFunction&,
        at::ObserverContext* context
    ) {
        auto* counter_context =
            dynamic_cast<NativeCounterObserverContext*>(
                context
            );

        if (
            counter_context == nullptr ||
            !counter_context->entered_canonical_scope()
        ) {
            return;
        }

        NativeCounterProbe* probe =
            counter_context->probe();

        if (
            probe != nullptr &&
            !probe->canonical_scope_stack_.empty()
        ) {
            probe->canonical_scope_stack_.pop_back();
        }
    }

    bool record_function(
        const at::RecordFunction& function,
        bool suppress_flop_calculation
    ) {
        const char* raw_name = function.name();

        const std::string operator_name =
            raw_name != nullptr
                ? std::string(raw_name)
                : std::string("<unnamed>");

        event_count_.fetch_add(
            1,
            std::memory_order_relaxed
        );

        ++operator_counts_[operator_name];

        try {
            operator_input_signatures_[operator_name] =
                describe_inputs(function);
        }
        catch (const std::exception& error) {
            operator_input_signatures_[operator_name] =
                std::string("<input inspection failed: ") +
                error.what() +
                ">";
        }
        catch (...) {
            operator_input_signatures_[operator_name] =
                "<input inspection failed>";
        }

        if (suppress_flop_calculation) {
            return false;
        }

        try {
            const FlopCalculator calculator =
                OperatorRegistry::instance().find(
                    operator_name
                );

            if (calculator == nullptr) {
                return false;
            }

            const std::uint64_t flops =
                calculator(function);

            if (flops == 0) {
                return false;
            }

            const std::uint64_t bit_width =
                get_effective_bit_width(function);

            const std::uint64_t bops =
                checked_multiply(
                    flops,
                    bit_width
                );

            operator_flops_[operator_name] =
                checked_add(
                    operator_flops_[operator_name],
                    flops
                );

            operator_bops_[operator_name] =
                checked_add(
                    operator_bops_[operator_name],
                    bops
                );

            add_flops(flops);
            add_bops(bops);

            return true;
        }
        catch (const std::exception& error) {
            operator_errors_[operator_name] =
                error.what();
        }
        catch (...) {
            operator_errors_[operator_name] =
                "Unknown FLOP calculation error.";
        }

        return false;
    }

    void stop_noexcept() noexcept {
        if (!active_) {
            return;
        }

        try {
            if (callback_handle_ != at::INVALID_CALLBACK_HANDLE) {
                at::removeCallback(callback_handle_);
            }
        }
        catch (...) {
        }

        callback_handle_ = at::INVALID_CALLBACK_HANDLE;
        paused_.store(false, std::memory_order_relaxed);
        canonical_scope_stack_.clear();
        active_ = false;

        if (tls_active_probe == this) {
            tls_active_probe = nullptr;
        }
    }

private:
    bool active_ = false;
    std::atomic<bool> paused_{false};

    at::CallbackHandle callback_handle_ =
        at::INVALID_CALLBACK_HANDLE;

    std::vector<CanonicalScopeState>
        canonical_scope_stack_;

    std::atomic<std::uint64_t> event_count_{0};
    std::atomic<std::uint64_t> total_flops_{0};
    std::atomic<std::uint64_t> total_bops_{0};

    std::unordered_map<std::string, std::uint64_t>
        operator_counts_;

    std::unordered_map<std::string, std::string>
        operator_input_signatures_;

    std::unordered_map<std::string, std::uint64_t>
        operator_flops_;

    std::unordered_map<std::string, std::uint64_t>
        operator_bops_;

    std::unordered_map<std::string, std::string>
        operator_errors_;
};

void bind_native_counter(py::module_& module) {
    module.doc() =
        "Native FLOP and BOP counter for PyTorch ATen operators.";

    py::class_<NativeCounterProbe>(
        module,
        "NativeCounterProbe"
    )
        .def(py::init<>())
        .def("start", &NativeCounterProbe::start)
        .def("stop", &NativeCounterProbe::stop)
        .def("reset", &NativeCounterProbe::reset)
        .def("pause", &NativeCounterProbe::pause)
        .def("resume", &NativeCounterProbe::resume)
        .def("add_flops", &NativeCounterProbe::add_flops)
        .def("add_bops", &NativeCounterProbe::add_bops)
        .def_property_readonly("active", &NativeCounterProbe::active)
        .def_property_readonly("paused", &NativeCounterProbe::paused)
        .def_property_readonly("event_count", &NativeCounterProbe::event_count)
        .def_property_readonly("total_flops", &NativeCounterProbe::total_flops)
        .def_property_readonly("total_bops", &NativeCounterProbe::total_bops)
        .def_property_readonly("operator_counts", &NativeCounterProbe::operator_counts)
        .def_property_readonly(
            "operator_input_signatures",
            &NativeCounterProbe::operator_input_signatures
        )
        .def_property_readonly("operator_flops", &NativeCounterProbe::operator_flops)
        .def_property_readonly("operator_bops", &NativeCounterProbe::operator_bops)
        .def_property_readonly("operator_errors", &NativeCounterProbe::operator_errors)
        .def_property_readonly(
            "supported_operators",
            &NativeCounterProbe::supported_operators
        )
        .def(
            "__enter__",
            [](NativeCounterProbe& self) -> NativeCounterProbe& {
                self.start();
                return self;
            },
            py::return_value_policy::reference_internal
        )
        .def(
            "__exit__",
            [](
                NativeCounterProbe& self,
                const py::object&,
                const py::object&,
                const py::object&
            ) {
                self.stop();
                return false;
            }
        );
}

}  // namespace floppy::native
