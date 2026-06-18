#include <pybind11/pybind11.h>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <cctype>

namespace py = pybind11;

static bool is_tensor(const py::handle& obj) {
    return py::hasattr(obj, "numel") && py::hasattr(obj, "dtype") && py::hasattr(obj, "dim");
}

static long long get_numel(const py::handle& obj) {
    if (obj.is_none()) return 0;

    if (is_tensor(obj)) {
        return obj.attr("numel")().cast<long long>();
    }

    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        long long total = 0;
        for (auto item : obj) total += get_numel(item);
        return total;
    }

    return 0;
}

static py::object get_first_tensor(const py::handle& obj) {
    if (obj.is_none()) return py::none();

    if (is_tensor(obj)) {
        return py::reinterpret_borrow<py::object>(obj);
    }

    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        for (auto item : obj) {
            py::object found = get_first_tensor(item);
            if (!found.is_none()) return found;
        }
    }

    if (py::isinstance<py::dict>(obj)) {
        py::dict d = py::reinterpret_borrow<py::dict>(obj);
        for (auto item : d) {
            py::object found = get_first_tensor(item.second);
            if (!found.is_none()) return found;
        }
    }

    return py::none();
}

static int get_effective_bit_width(const py::handle& obj) {
    py::object tensor = get_first_tensor(obj);

    if (tensor.is_none() || !py::hasattr(tensor, "dtype")) {
        return 32;
    }

    std::string dtype = py::str(tensor.attr("dtype"));

    if (dtype == "torch.float64" || dtype == "torch.int64" || dtype == "torch.complex128") return 64;
    if (dtype == "torch.float32" || dtype == "torch.int32" || dtype == "torch.complex64") return 32;
    if (dtype == "torch.float16" || dtype == "torch.bfloat16" || dtype == "torch.int16") return 16;
    if (dtype == "torch.int8" || dtype == "torch.uint8" || dtype == "torch.qint8") return 8;
    if (dtype == "torch.quint4x2" || dtype == "torch.int4") return 4;

    return 32;
}

static bool contains(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}

static bool in_vector(const std::string& value, const std::vector<std::string>& values) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

static std::string to_lower(std::string s) {
    std::transform(
        s.begin(),
        s.end(),
        s.begin(),
        [](unsigned char c) { return std::tolower(c); }
    );
    return s;
}

static std::string get_base_op(const std::string& func_name) {
    std::size_t pos = func_name.find(".");
    if (pos != std::string::npos && pos + 1 < func_name.size()) {
        std::string rest = func_name.substr(pos + 1);
        std::size_t dot2 = rest.find(".");
        if (dot2 != std::string::npos) return rest.substr(0, dot2);
        return rest;
    }

    const std::string prefix = "aten.";
    if (func_name.rfind(prefix, 0) == 0) {
        return func_name.substr(prefix.size());
    }

    return func_name;
}

static long long dim(const py::handle& tensor) {
    return tensor.attr("dim")().cast<long long>();
}

static long long shape_at(const py::handle& tensor, long long index) {
    return tensor.attr("shape").attr("__getitem__")(index).cast<long long>();
}

static std::pair<long long, long long> make_result(long long flops, const py::handle& ref_obj) {
    if (flops <= 0) return {0, 0};

    int bit_width = get_effective_bit_width(ref_obj);
    return {flops, flops * bit_width};
}

std::pair<long long, long long> compute_operation(
    py::object func,
    py::object args,
    py::object out
) {
    std::string func_name = to_lower(py::str(func).cast<std::string>());

    long long out_elements = get_numel(out);
    long long in_elements = 0;

    if (py::isinstance<py::tuple>(args) || py::isinstance<py::list>(args)) {
        if (py::len(args) > 0) {
            in_elements = get_numel(args.attr("__getitem__")(0));
        }
    }

    std::string base_op = get_base_op(func_name);

    static const std::vector<std::string> elementwise_ops = {
        "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_",
        "exp", "exp_", "log", "log_", "pow", "pow_", "neg", "neg_", "abs", "abs_",
        "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_",
        "sqrt", "sqrt_", "rsqrt", "rsqrt_", "gelu", "gelu_", "silu", "silu_", "mish", "mish_",
        "_foreach_add", "_foreach_add_", "_foreach_sub", "_foreach_sub_",
        "_foreach_mul", "_foreach_mul_", "_foreach_div", "_foreach_div_",
        "_foreach_sqrt", "_foreach_sqrt_", "_foreach_exp", "_foreach_exp_",
        "_foreach_neg", "_foreach_neg_"
    };

    static const std::vector<std::string> complex_optimizer_ops = {
        "addcdiv", "addcdiv_", "_foreach_addcdiv", "_foreach_addcdiv_",
        "addcmul", "addcmul_", "_foreach_addcmul", "_foreach_addcmul_"
    };

    if (in_vector(base_op, elementwise_ops)) {
        long long elements = out_elements > 0 ? out_elements : in_elements;
        return make_result(elements, args);
    }

    if (in_vector(base_op, complex_optimizer_ops)) {
        long long elements = out_elements > 0 ? out_elements : in_elements;
        return make_result(elements * 3, args);
    }

    if (contains(func_name, "aten.mm") || contains(func_name, "aten.addmm")) {
        py::object mat1 = contains(func_name, "addmm") ? args.attr("__getitem__")(1) : args.attr("__getitem__")(0);
        py::object mat2 = contains(func_name, "addmm") ? args.attr("__getitem__")(2) : args.attr("__getitem__")(1);

        if (is_tensor(mat1) && is_tensor(mat2) && dim(mat1) == 2) {
            long long m = shape_at(mat1, 0);
            long long k = shape_at(mat1, 1);
            long long n = shape_at(mat2, 1);
            return make_result(2LL * m * n * k, mat1);
        }
    }

    if (contains(func_name, "aten.bmm") || contains(func_name, "aten.baddbmm")) {
        py::object mat1 = contains(func_name, "baddbmm") ? args.attr("__getitem__")(1) : args.attr("__getitem__")(0);
        py::object mat2 = contains(func_name, "baddbmm") ? args.attr("__getitem__")(2) : args.attr("__getitem__")(1);

        if (is_tensor(mat1) && is_tensor(mat2) && dim(mat1) == 3) {
            long long b = shape_at(mat1, 0);
            long long m = shape_at(mat1, 1);
            long long k = shape_at(mat1, 2);
            long long n = shape_at(mat2, 2);
            return make_result(2LL * b * m * n * k, mat1);
        }
    }

    if (contains(func_name, "aten.embedding")) {
        py::object weight = args.attr("__getitem__")(0);
        py::object indices = args.attr("__getitem__")(1);

        if (is_tensor(weight) && is_tensor(indices)) {
            long long flops = get_numel(indices) * shape_at(weight, -1);
            return make_result(flops, weight);
        }
    }

    if (
        contains(func_name, "aten.layer_norm") ||
        contains(func_name, "aten.native_layer_norm") ||
        contains(func_name, "aten.rms_norm")
    ) {
        py::object x = args.attr("__getitem__")(0);
        if (is_tensor(x)) return make_result(4LL * get_numel(x), x);
    }

    if (
        contains(func_name, "aten._softmax") ||
        contains(func_name, "aten.softmax") ||
        contains(func_name, "aten._log_softmax")
    ) {
        return make_result(in_elements * 3, args);
    }

    if (
        contains(func_name, "aten.nll_loss") ||
        contains(func_name, "aten.cross_entropy_loss")
    ) {
        return make_result(in_elements, args);
    }

    if (contains(func_name, "aten.convolution") || contains(func_name, "aten.conv")) {
        py::object weight = args.attr("__getitem__")(1);

        if (is_tensor(weight)) {
            long long weight_numel = get_numel(weight);
            long long out_channels = shape_at(weight, 0);
            double flops_per_element = 2.0 * (
                static_cast<double>(weight_numel) / static_cast<double>(out_channels)
            );
            long long flops = static_cast<long long>(out_elements * flops_per_element);
            return make_result(flops, weight);
        }
    }

    if (contains(func_name, "aten._scaled_dot_product")) {
        py::object q = args.attr("__getitem__")(0);
        py::object k = args.attr("__getitem__")(1);
        py::object v = args.attr("__getitem__")(2);

        if (is_tensor(q) && is_tensor(k) && is_tensor(v)) {
            long long batch_size = shape_at(q, 0);
            long long num_heads = shape_at(q, 1);
            long long seq_len_q = shape_at(q, 2);
            long long seq_len_k = shape_at(k, 2);
            long long head_dim = shape_at(q, 3);

            long long flops_qk = 2LL * batch_size * num_heads * seq_len_q * seq_len_k * head_dim;
            long long flops_softmax = 3LL * batch_size * num_heads * seq_len_q * seq_len_k;
            long long flops_v = 2LL * batch_size * num_heads * seq_len_q * seq_len_k * head_dim;

            return make_result(flops_qk + flops_softmax + flops_v, q);
        }
    }

    if (
        contains(func_name, "aten.scatter") ||
        contains(func_name, "aten.scatter_add") ||
        contains(func_name, "aten.scatter_reduce") ||
        contains(func_name, "aten.index_add") ||
        contains(func_name, "aten.index_select") ||
        contains(func_name, "aten.gather")
    ) {
        return make_result(out_elements, args);
    }

    if (contains(func_name, "aten.bincount")) {
        return make_result(in_elements, args);
    }

    if (
        contains(func_name, "aten.max_pool") ||
        contains(func_name, "aten.avg_pool") ||
        contains(func_name, "aten.adaptive_avg_pool")
    ) {
        return make_result(out_elements, args);
    }

    if (contains(func_name, "aten.upsample") || contains(func_name, "aten._upsample")) {
        return make_result(out_elements * 4, args);
    }

    if (
        contains(func_name, "aten.topk") ||
        contains(func_name, "aten.sort") ||
        contains(func_name, "aten.argsort")
    ) {
        if (in_elements > 0) {
            long long flops = static_cast<long long>(
                static_cast<double>(in_elements) *
                std::log2(static_cast<double>(std::max<long long>(2, in_elements)))
            );
            return make_result(flops, args);
        }
    }

    if (
        contains(func_name, "aten.lstm") ||
        contains(func_name, "aten.gru") ||
        contains(func_name, "aten.rnn")
    ) {
        py::object x = args.attr("__getitem__")(0);

        if (is_tensor(x) && dim(x) >= 2) {
            return make_result(get_numel(x) * 8, args);
        }
    }

    return {0, 0};
}

PYBIND11_MODULE(_flop_counter_core, m) {
    m.doc() = "C++ core for FLOPpy UniversalFlopCounter";

    m.def(
        "compute_operation",
        &compute_operation,
        py::arg("func"),
        py::arg("args"),
        py::arg("out")
    );
}