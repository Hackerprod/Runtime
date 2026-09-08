#include "mm_runtime.h"

#include "mm_kernels.h"
#include "mm_parallel.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kVersion = 1;
constexpr uint32_t kTiedOnly = 1;
constexpr size_t kMaxFileBytes = static_cast<size_t>(2) * 1024 * 1024 * 1024;
constexpr size_t kMaxRuntimeTablesBytes = static_cast<size_t>(1) * 1024 * 1024 * 1024;

struct Config {
    uint32_t version = 0;
    uint32_t vocab = 0, hidden = 0, layers = 0, q_heads = 0, kv_heads = 0;
    uint32_t head_dim = 0, intermediate = 0, max_positions = 0, eos = 0;
    uint32_t flags = 0, tensor_count = 0;
    float rms_eps = 0.0f, rope_theta = 0.0f;
};

struct Tensor {
    uint32_t dtype = 0;
    std::vector<uint32_t> shape;
    std::vector<float> f32;
    std::vector<uint8_t> q4;
    std::vector<float> scales;

    size_t rows() const { return shape.size() == 1 ? 1 : shape[0]; }
    size_t cols() const { return shape.size() == 1 ? shape[0] : shape[1]; }
};

struct LayerWeights {
    const Tensor* input_norm = nullptr;
    const Tensor* post_attention_norm = nullptr;
    const Tensor* q_proj = nullptr;
    const Tensor* k_proj = nullptr;
    const Tensor* v_proj = nullptr;
    const Tensor* o_proj = nullptr;
    const Tensor* q_norm = nullptr;
    const Tensor* k_norm = nullptr;
    const Tensor* gate_proj = nullptr;
    const Tensor* up_proj = nullptr;
    const Tensor* down_proj = nullptr;
};

struct Expected {
    std::vector<uint32_t> shape;
    bool embedding = false;
};

static size_t checked_mul(size_t a, size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        throw std::runtime_error(std::string("size overflow in ") + what);
    }
    return a * b;
}

static size_t checked_add(size_t a, size_t b, const char* what) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        throw std::runtime_error(std::string("size overflow in ") + what);
    }
    return a + b;
}

static void set_error(char* out, size_t cap, const std::string& message) {
    if (out == nullptr || cap == 0) return;
    const size_t n = std::min(cap - 1, message.size());
    std::memcpy(out, message.data(), n);
    out[n] = '\0';
}

class Reader {
public:
    Reader(const std::vector<uint8_t>& bytes, size_t limit) : bytes_(bytes), limit_(limit) {}

    uint32_t u32() {
        const uint8_t* p = take(4);
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
    }

    uint64_t u64() {
        const uint8_t* p = take(8);
        uint64_t result = 0;
        for (unsigned i = 0; i < 8; ++i) result |= static_cast<uint64_t>(p[i]) << (i * 8);
        return result;
    }

    float f32() {
        const uint32_t bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string ascii(size_t n) {
        if (n > 512) throw std::runtime_error("tensor name is too long");
        const uint8_t* p = take(n);
        for (size_t i = 0; i < n; ++i) {
            if (p[i] < 0x20 || p[i] > 0x7e) throw std::runtime_error("tensor name is not ASCII");
        }
        return std::string(reinterpret_cast<const char*>(p), n);
    }

    const uint8_t* raw(size_t n) { return take(n); }
    size_t position() const { return position_; }
    size_t remaining() const { return limit_ - position_; }

private:
    const uint8_t* take(size_t n) {
        if (n > remaining()) throw std::runtime_error("truncated tensor file");
        const uint8_t* result = bytes_.data() + position_;
        position_ += n;
        return result;
    }

    const std::vector<uint8_t>& bytes_;
    size_t limit_;
    size_t position_ = 0;
};

static uint32_t crc32(const uint8_t* data, size_t size) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (unsigned bit = 0; bit < 8; ++bit) {
                const uint32_t mask = 0u - (value & 1u);
                value = (value >> 1) ^ (0xedb88320u & mask);
            }
            values[i] = value;
        }
        return values;
    }();
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

static void validate_config(const Config& c) {
    if (c.version != kVersion) throw std::runtime_error("unsupported file version");
    if (c.flags != kTiedOnly) throw std::runtime_error("unsupported model flags");
    if (c.vocab == 0 || c.hidden == 0 || c.layers == 0 || c.q_heads == 0 ||
        c.kv_heads == 0 || c.head_dim == 0 || c.intermediate == 0 || c.max_positions == 0) {
        throw std::runtime_error("model dimensions must be positive");
    }
    if (c.hidden != c.q_heads * c.head_dim) throw std::runtime_error("hidden size does not match query heads");
    if ((c.head_dim & 1u) != 0) throw std::runtime_error("head dimension must be even for RoPE");
    if (c.q_heads % c.kv_heads != 0) throw std::runtime_error("query heads must be divisible by KV heads");
    if (c.eos >= c.vocab) throw std::runtime_error("EOS token is outside vocabulary");
    if (!std::isfinite(c.rms_eps) || c.rms_eps <= 0.0f ||
        !std::isfinite(c.rope_theta) || c.rope_theta <= 0.0f) {
        throw std::runtime_error("invalid normalization or RoPE parameter");
    }
    // Keep malformed headers from causing unbounded allocation before tensor validation.
    if (c.vocab > 10'000'000 || c.hidden > 1'000'000 || c.layers > 1'000 ||
        c.q_heads > 16'384 || c.kv_heads > 16'384 || c.head_dim > 16'384 ||
        c.intermediate > 4'000'000 || c.max_positions > 1'000'000'000) {
        throw std::runtime_error("model dimensions exceed safety bounds");
    }
}

static std::unordered_map<std::string, Expected> expected_tensors(const Config& c) {
    std::unordered_map<std::string, Expected> expected;
    expected.reserve(static_cast<size_t>(c.layers) * 11 + 2);
    expected.emplace("model.embed_tokens.weight", Expected{{c.vocab, c.hidden}, true});
    expected.emplace("model.norm.weight", Expected{{c.hidden}, false});
    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        const std::string p = "model.layers." + std::to_string(layer);
        expected.emplace(p + ".input_layernorm.weight", Expected{{c.hidden}, false});
        expected.emplace(p + ".post_attention_layernorm.weight", Expected{{c.hidden}, false});
        expected.emplace(p + ".self_attn.q_proj.weight", Expected{{c.q_heads * c.head_dim, c.hidden}, false});
        expected.emplace(p + ".self_attn.k_proj.weight", Expected{{c.kv_heads * c.head_dim, c.hidden}, false});
        expected.emplace(p + ".self_attn.v_proj.weight", Expected{{c.kv_heads * c.head_dim, c.hidden}, false});
        expected.emplace(p + ".self_attn.o_proj.weight", Expected{{c.hidden, c.q_heads * c.head_dim}, false});
        expected.emplace(p + ".self_attn.q_norm.weight", Expected{{c.head_dim}, false});
        expected.emplace(p + ".self_attn.k_norm.weight", Expected{{c.head_dim}, false});
        expected.emplace(p + ".mlp.gate_proj.weight", Expected{{c.intermediate, c.hidden}, false});
        expected.emplace(p + ".mlp.up_proj.weight", Expected{{c.intermediate, c.hidden}, false});
        expected.emplace(p + ".mlp.down_proj.weight", Expected{{c.hidden, c.intermediate}, false});
    }
    return expected;
}

static std::unordered_map<std::string, Tensor> load_tensors(const std::vector<uint8_t>& bytes, Config& config) {
    if (bytes.size() < 8 + 12 * 4 + 2 * 4 + 4) throw std::runtime_error("tensor file is too small");
    if (std::memcmp(bytes.data(), "MMCPU001", 8) != 0) throw std::runtime_error("invalid MMCPU001 magic");
    if (bytes.size() > kMaxFileBytes) throw std::runtime_error("tensor file exceeds safety bound");

    const size_t payload_size = bytes.size() - 4;
    Reader reader(bytes, payload_size);
    reader.raw(8); // magic
    config.version = reader.u32();
    config.vocab = reader.u32();
    config.hidden = reader.u32();
    config.layers = reader.u32();
    config.q_heads = reader.u32();
    config.kv_heads = reader.u32();
    config.head_dim = reader.u32();
    config.intermediate = reader.u32();
    config.max_positions = reader.u32();
    config.eos = reader.u32();
    config.flags = reader.u32();
    config.tensor_count = reader.u32();
    config.rms_eps = reader.f32();
    config.rope_theta = reader.f32();
    validate_config(config);

    const auto expected = expected_tensors(config);
    if (config.tensor_count != expected.size()) throw std::runtime_error("unexpected tensor count");
    std::unordered_map<std::string, Tensor> tensors;
    std::vector<LayerWeights> layers;
    tensors.reserve(expected.size());
    for (uint32_t i = 0; i < config.tensor_count; ++i) {
        const uint32_t name_len = reader.u32();
        const std::string name = reader.ascii(name_len);
        const auto expected_it = expected.find(name);
        if (expected_it == expected.end()) throw std::runtime_error("unexpected tensor: " + name);
        if (tensors.find(name) != tensors.end()) throw std::runtime_error("duplicate tensor: " + name);

        Tensor tensor;
        tensor.dtype = reader.u32();
        const uint32_t rank = reader.u32();
        if (rank == 0 || rank > 2) throw std::runtime_error("unsupported tensor rank");
        tensor.shape.resize(rank);
        size_t elements = 1;
        for (uint32_t& dim : tensor.shape) {
            dim = reader.u32();
            if (dim == 0) throw std::runtime_error("zero tensor dimension");
            elements = checked_mul(elements, dim, "tensor elements");
        }
        const uint32_t group = reader.u32();
        const uint64_t data_bytes64 = reader.u64();
        const uint64_t scale_count64 = reader.u64();
        if (data_bytes64 > reader.remaining() || scale_count64 > reader.remaining() / sizeof(float)) {
            throw std::runtime_error("tensor payload exceeds file bounds");
        }
        const size_t data_bytes = static_cast<size_t>(data_bytes64);
        const size_t scale_count = static_cast<size_t>(scale_count64);
        if (tensor.shape != expected_it->second.shape) throw std::runtime_error("tensor shape mismatch: " + name);
        if (expected_it->second.embedding && tensor.dtype != 0) throw std::runtime_error("embedding must be FP32");
        if (tensor.shape.size() == 1 && tensor.dtype != 0) throw std::runtime_error("norm tensor must be FP32");
        if (tensor.dtype != 0 && tensor.dtype != 1) throw std::runtime_error("unsupported tensor dtype");
        if (tensor.dtype == 0) {
            if (group != 0 || data_bytes != checked_mul(elements, sizeof(float), "FP32 bytes") || scale_count != 0) {
                throw std::runtime_error("invalid FP32 tensor payload: " + name);
            }
            tensor.f32.resize(elements);
            for (float& value : tensor.f32) {
                value = reader.f32();
                if (!std::isfinite(value)) throw std::runtime_error("non-finite FP32 tensor value: " + name);
            }
        } else {
            if (tensor.shape.size() != 2 || group != 32) throw std::runtime_error("invalid Q4 tensor group");
            const size_t rows = tensor.shape[0], cols = tensor.shape[1];
            const size_t row_bytes = (cols + 1) / 2;
            const size_t groups = (cols + 31) / 32;
            const size_t expected_bytes = checked_mul(rows, row_bytes, "Q4 bytes");
            const size_t expected_scales = checked_mul(rows, groups, "Q4 scales");
            if (data_bytes != expected_bytes || scale_count != expected_scales) {
                throw std::runtime_error("invalid Q4 tensor payload: " + name);
            }
            const uint8_t* raw = reader.raw(data_bytes);
            tensor.q4.assign(raw, raw + data_bytes);
            tensor.scales.resize(scale_count);
            for (float& scale : tensor.scales) {
                scale = reader.f32();
                if (!std::isfinite(scale) || scale <= 0.0f) throw std::runtime_error("invalid Q4 scale: " + name);
            }
        }
        tensors.emplace(name, std::move(tensor));
    }
    if (reader.position() != payload_size) throw std::runtime_error("trailing or missing tensor payload");

    Reader crc_reader(bytes, bytes.size());
    crc_reader.raw(bytes.size() - 4);
    const uint32_t stored_crc = crc_reader.u32();
    const uint32_t actual_crc = crc32(bytes.data(), bytes.size() - 4);
    if (stored_crc != actual_crc) throw std::runtime_error("CRC32 mismatch");
    if (tensors.size() != expected.size()) throw std::runtime_error("missing tensor");
    for (const auto& item : expected) {
        if (tensors.find(item.first) == tensors.end()) throw std::runtime_error("missing tensor: " + item.first);
    }
    return tensors;
}

struct Runtime {
    enum class ProfileOp : std::uint32_t { Qkv, Attention, Output, Ffn, Vocab, Remaining };
    MmRuntimeStats stats{};
    bool profile_enabled = false;
    bool selective_logits = false;
    bool v_blocked_attention = false;
    bool ffn_row4 = false;
    bool gqa_k_shared = false;
    bool logits_valid = false;
    std::uint64_t cache_epoch = 0;
    Config config;
    uint32_t max_context = 0;
    mm::KernelMode kernel_mode = mm::KernelMode::Scalar;
    mm::ParallelTeam parallel;
    std::string backend;
    std::unordered_map<std::string, Tensor> tensors;
    std::vector<LayerWeights> layers;
    std::vector<float> cache_k;
    std::vector<float> cache_v;
    uint32_t position = 0;

    std::vector<float> hidden, residual, normed, projection;
    std::vector<float> q, k, v, q_rot, k_rot, attention;
    std::vector<float> intermediate, ffn_up, logits, scores;
    std::vector<float> rope_cos, rope_sin;

    Runtime(Config c, std::unordered_map<std::string, Tensor> t, uint32_t context, int mode)
        : config(c), max_context(context), kernel_mode(static_cast<mm::KernelMode>(mode)), tensors(std::move(t)) {
        layers.resize(config.layers);
        for (uint32_t layer = 0; layer < config.layers; ++layer) {
            const std::string p = "model.layers." + std::to_string(layer);
            LayerWeights& w = layers[layer];
            w.input_norm = &tensor(p + ".input_layernorm.weight");
            w.post_attention_norm = &tensor(p + ".post_attention_layernorm.weight");
            w.q_proj = &tensor(p + ".self_attn.q_proj.weight");
            w.k_proj = &tensor(p + ".self_attn.k_proj.weight");
            w.v_proj = &tensor(p + ".self_attn.v_proj.weight");
            w.o_proj = &tensor(p + ".self_attn.o_proj.weight");
            w.q_norm = &tensor(p + ".self_attn.q_norm.weight");
            w.k_norm = &tensor(p + ".self_attn.k_norm.weight");
            w.gate_proj = &tensor(p + ".mlp.gate_proj.weight");
            w.up_proj = &tensor(p + ".mlp.up_proj.weight");
            w.down_proj = &tensor(p + ".mlp.down_proj.weight");
        }
        const size_t kv_size = checked_mul(checked_mul(checked_mul(config.layers, max_context, "KV cache"), config.kv_heads, "KV cache"), config.head_dim, "KV cache");
        const size_t rope_half = config.head_dim / 2;
        const size_t rope_values = checked_mul(max_context, rope_half, "RoPE table");
        const size_t cache_bytes = checked_mul(checked_mul(kv_size, sizeof(float), "KV cache bytes"), 2, "KV cache bytes");
        const size_t rope_bytes = checked_mul(checked_mul(rope_values, sizeof(float), "RoPE table bytes"), 2, "RoPE table bytes");
        const size_t score_values = checked_mul(config.q_heads, max_context, "attention scratch");
        const size_t score_bytes = checked_mul(score_values, sizeof(float), "attention scratch bytes");
        const size_t table_bytes = checked_add(checked_add(cache_bytes, rope_bytes, "runtime tables"), score_bytes, "runtime tables");
        if (table_bytes > kMaxRuntimeTablesBytes) throw std::runtime_error("runtime cache/tables exceed safety bound");
        cache_k.resize(kv_size);
        cache_v.resize(kv_size);
        rope_cos.resize(rope_values);
        rope_sin.resize(rope_values);
        for (uint32_t pos = 0; pos < max_context; ++pos) {
            for (uint32_t i = 0; i < config.head_dim / 2; ++i) {
                const float angle = static_cast<float>(pos) * std::pow(config.rope_theta,
                    -(2.0f * i) / static_cast<float>(config.head_dim));
                rope_cos[static_cast<size_t>(pos) * rope_half + i] = std::cos(angle);
                rope_sin[static_cast<size_t>(pos) * rope_half + i] = std::sin(angle);
            }
        }
        hidden.resize(config.hidden);
        residual.resize(config.hidden);
        normed.resize(config.hidden);
        projection.resize(config.hidden);
        q.resize(static_cast<size_t>(config.q_heads) * config.head_dim);
        k.resize(static_cast<size_t>(config.kv_heads) * config.head_dim);
        v.resize(static_cast<size_t>(config.kv_heads) * config.head_dim);
        q_rot.resize(q.size());
        k_rot.resize(k.size());
        attention.resize(q.size());
        intermediate.resize(config.intermediate);
        ffn_up.resize(config.intermediate);
        logits.resize(config.vocab);
        scores.resize(score_values);
        const char* name = mm::kernel_name(kernel_mode);
        backend = name == nullptr ? "unknown" : name;
    }

    const Tensor& tensor(const std::string& name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("internal missing tensor: " + name);
        return it->second;
    }

    size_t cache_offset(uint32_t layer, uint32_t pos, uint32_t head, uint32_t dim) const {
        return (((static_cast<size_t>(layer) * max_context + pos) * config.kv_heads + head) * config.head_dim + dim);
    }

    static void rms_norm(const std::vector<float>& input, const Tensor& weight, float eps, std::vector<float>& output) {
        float sum = 0.0f;
        for (float value : input) sum += value * value;
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(input.size()) + eps);
        for (size_t i = 0; i < input.size(); ++i) output[i] = input[i] * inv * weight.f32[i];
    }

    static void rms_norm_ptr(const float* input, const Tensor& weight, size_t size,
                             float eps, float* output) {
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) sum += input[i] * input[i];
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);
        for (size_t i = 0; i < size; ++i) output[i] = input[i] * inv * weight.f32[i];
    }

    void gemv(const Tensor& weight, const float* input, float* output,
              ProfileOp op = ProfileOp::Remaining, bool use_ffn_row4 = false) {
        const uint32_t rows = static_cast<uint32_t>(weight.rows());
        const uint32_t cols = static_cast<uint32_t>(weight.cols());
        const auto started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (weight.dtype == 0 && use_ffn_row4 && ffn_row4) parallel.gemv_f32_row4(weight.f32.data(), input, output, rows, cols, kernel_mode);
        else if (weight.dtype == 0) parallel.gemv_f32(weight.f32.data(), input, output, rows, cols, kernel_mode);
        else parallel.gemv_q4(weight.q4.data(), weight.scales.data(), input, output, rows, cols, kernel_mode);
        if (profile_enabled) {
            const auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
            switch (op) {
            case ProfileOp::Qkv: ++stats.qkv_calls; stats.qkv_ns += ns; break;
            case ProfileOp::Output: ++stats.output_projection_calls; stats.output_projection_ns += ns; break;
            case ProfileOp::Ffn: ++stats.ffn_calls; stats.ffn_ns += ns; break;
            case ProfileOp::Vocab: ++stats.vocab_head_calls; stats.vocab_head_ns += ns; break;
            default: ++stats.remaining_ops_calls; stats.remaining_ops_ns += ns; break;
            }
        }
    }

    void rotate(uint32_t pos) {
        const size_t rope_base = static_cast<size_t>(pos) * (config.head_dim / 2);
        for (uint32_t h = 0; h < config.q_heads; ++h) {
            const size_t base = static_cast<size_t>(h) * config.head_dim;
            for (uint32_t i = 0; i < config.head_dim / 2; ++i) {
                const float c = rope_cos[rope_base + i], s = rope_sin[rope_base + i];
                const float first = q[base + i], second = q[base + i + config.head_dim / 2];
                q_rot[base + i] = first * c - second * s;
                q_rot[base + i + config.head_dim / 2] = second * c + first * s;
            }
        }
        for (uint32_t h = 0; h < config.kv_heads; ++h) {
            const size_t base = static_cast<size_t>(h) * config.head_dim;
            for (uint32_t i = 0; i < config.head_dim / 2; ++i) {
                const float c = rope_cos[rope_base + i], s = rope_sin[rope_base + i];
                const float first = k[base + i], second = k[base + i + config.head_dim / 2];
                k_rot[base + i] = first * c - second * s;
                k_rot[base + i + config.head_dim / 2] = second * c + first * s;
            }
        }
    }

    void step(int32_t token, bool emit_logits = true) {
        logits_valid = false;
        const auto step_started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto qkv_before = stats.qkv_ns; const auto attn_before = stats.attention_kv_ns;
        const auto output_before = stats.output_projection_ns; const auto ffn_before = stats.ffn_ns;
        const auto vocab_before = stats.vocab_head_ns;
        const Tensor& embedding = tensor("model.embed_tokens.weight");
        std::copy_n(embedding.f32.data() + static_cast<size_t>(token) * config.hidden, config.hidden, hidden.data());
        for (uint32_t layer = 0; layer < config.layers; ++layer) {
            const LayerWeights& w = layers[layer];
            if (profile_enabled) ++stats.remaining_ops_calls;
            std::copy(hidden.begin(), hidden.end(), residual.begin());
            rms_norm(hidden, *w.input_norm, config.rms_eps, normed);
            gemv(*w.q_proj, normed.data(), q.data(), ProfileOp::Qkv);
            gemv(*w.k_proj, normed.data(), k.data(), ProfileOp::Qkv);
            gemv(*w.v_proj, normed.data(), v.data(), ProfileOp::Qkv);
            for (uint32_t h = 0; h < config.q_heads; ++h) {
                const size_t base = static_cast<size_t>(h) * config.head_dim;
                rms_norm_ptr(q.data() + base, *w.q_norm, config.head_dim,
                             config.rms_eps, q_rot.data() + base);
            }
            for (uint32_t h = 0; h < config.kv_heads; ++h) {
                const size_t base = static_cast<size_t>(h) * config.head_dim;
                rms_norm_ptr(k.data() + base, *w.k_norm, config.head_dim,
                             config.rms_eps, k_rot.data() + base);
            }
            std::copy(q_rot.begin(), q_rot.end(), q.begin());
            std::copy(k_rot.begin(), k_rot.end(), k.begin());
            rotate(position);
            if (!emit_logits) {
                for (float value : k_rot) if (!std::isfinite(value)) throw std::runtime_error("non-finite key state");
                for (float value : v) if (!std::isfinite(value)) throw std::runtime_error("non-finite value state");
            }
            std::copy(q_rot.begin(), q_rot.end(), q.begin());
            std::copy(k_rot.begin(), k_rot.end(), k.begin());
            for (uint32_t h = 0; h < config.kv_heads; ++h) {
                for (uint32_t d = 0; d < config.head_dim; ++d) {
                    cache_k[cache_offset(layer, position, h, d)] = k[static_cast<size_t>(h) * config.head_dim + d];
                    cache_v[cache_offset(layer, position, h, d)] = v[static_cast<size_t>(h) * config.head_dim + d];
                }
            }
            const auto attention_started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            std::fill(attention.begin(), attention.end(), 0.0f);
            const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(config.head_dim));
            const uint32_t total = position + 1;
            const auto finish_attention_head = [&](uint32_t h, uint32_t kvh, float max_score) {
                float denom = 0.0f;
                for (uint32_t t = 0; t < total; ++t) {
                    const size_t index = static_cast<size_t>(h) * max_context + t;
                    scores[index] = std::exp(scores[index] - max_score);
                    denom += scores[index];
                }
                for (uint32_t t = 0; t < total; ++t) scores[static_cast<size_t>(h) * max_context + t] /= denom;
                float* const head_attention = attention.data() + static_cast<size_t>(h) * config.head_dim;
                if (!v_blocked_attention) {
                    for (uint32_t d = 0; d < config.head_dim; ++d) {
                        float value = 0.0f;
                        for (uint32_t t = 0; t < total; ++t) {
                            const float product = scores[static_cast<size_t>(h) * max_context + t] * cache_v[cache_offset(layer, t, kvh, d)];
                            value = value + product;
                        }
                        head_attention[d] = value;
                    }
                } else {
                    constexpr uint32_t block_dims = 16;
                    for (uint32_t block = 0; block < config.head_dim; block += block_dims) {
                        const uint32_t end = std::min(block + block_dims, config.head_dim);
                        for (uint32_t d = block; d < end; ++d) head_attention[d] = 0.0f;
                        for (uint32_t t = 0; t < total; ++t) {
                            const float score = scores[static_cast<size_t>(h) * max_context + t];
                            const float* const value_row = cache_v.data() + cache_offset(layer, t, kvh, block);
                            for (uint32_t d = block; d < end; ++d) {
                                const float product = score * value_row[d - block];
                                head_attention[d] = head_attention[d] + product;
                            }
                        }
                    }
                }
            };
            const uint32_t q_per_kv = config.q_heads / config.kv_heads;
            if (gqa_k_shared && q_per_kv == 2) {
                for (uint32_t h = 0; h < config.q_heads; h += 2) {
                    const uint32_t kvh = h / q_per_kv;
                    float max_score0 = -std::numeric_limits<float>::infinity();
                    float max_score1 = -std::numeric_limits<float>::infinity();
                    const auto q0 = q.data() + static_cast<size_t>(h) * config.head_dim;
                    const auto q1 = q.data() + static_cast<size_t>(h + 1) * config.head_dim;
                    const auto qk_started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    for (uint32_t t = 0; t < total; ++t) {
                        float score0 = 0.0f;
                        float score1 = 0.0f;
                        const float* const key_row = cache_k.data() + cache_offset(layer, t, kvh, 0);
                        for (uint32_t d = 0; d < config.head_dim; ++d) {
                            const float key = key_row[d];
                            score0 += q0[d] * key;
                            score1 += q1[d] * key;
                        }
                        score0 *= inv_sqrt;
                        score1 *= inv_sqrt;
                        scores[static_cast<size_t>(h) * max_context + t] = score0;
                        scores[static_cast<size_t>(h + 1) * max_context + t] = score1;
                        max_score0 = std::max(max_score0, score0);
                        max_score1 = std::max(max_score1, score1);
                    }
                    if (profile_enabled) {
                        stats.attention_qk_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - qk_started).count());
                    }
                    finish_attention_head(h, kvh, max_score0);
                    finish_attention_head(h + 1, kvh, max_score1);
                }
            } else {
                for (uint32_t h = 0; h < config.q_heads; ++h) {
                    const uint32_t kvh = h / q_per_kv;
                    float max_score = -std::numeric_limits<float>::infinity();
                    const auto qk_started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    for (uint32_t t = 0; t < total; ++t) {
                        float score = 0.0f;
                        for (uint32_t d = 0; d < config.head_dim; ++d) {
                            score += q[static_cast<size_t>(h) * config.head_dim + d] * cache_k[cache_offset(layer, t, kvh, d)];
                        }
                        score *= inv_sqrt;
                        scores[static_cast<size_t>(h) * max_context + t] = score;
                        max_score = std::max(max_score, score);
                    }
                    if (profile_enabled) {
                        stats.attention_qk_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - qk_started).count());
                    }
                    finish_attention_head(h, kvh, max_score);
                }
            }
            if (profile_enabled) {
                stats.attention_kv_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - attention_started).count());
                ++stats.attention_kv_calls;
            }
            gemv(*w.o_proj, attention.data(), projection.data(), ProfileOp::Output);
            for (uint32_t d = 0; d < config.hidden; ++d) hidden[d] = residual[d] + projection[d];
            rms_norm(hidden, *w.post_attention_norm, config.rms_eps, normed);
            gemv(*w.gate_proj, normed.data(), intermediate.data(), ProfileOp::Ffn, true);
            gemv(*w.up_proj, normed.data(), ffn_up.data(), ProfileOp::Ffn, true);
            for (uint32_t d = 0; d < config.intermediate; ++d) {
                const float gate = intermediate[d];
                intermediate[d] = (gate / (1.0f + std::exp(-gate))) * ffn_up[d];
            }
            gemv(*w.down_proj, intermediate.data(), projection.data(), ProfileOp::Ffn, true);
            for (uint32_t d = 0; d < config.hidden; ++d) hidden[d] += projection[d];
        }
        rms_norm(hidden, tensor("model.norm.weight"), config.rms_eps, normed);
        for (float value : normed) if (!std::isfinite(value)) throw std::runtime_error("non-finite hidden state");
        if (emit_logits) {
            ++stats.lm_head_calls;
            gemv(tensor("model.embed_tokens.weight"), normed.data(), logits.data(), ProfileOp::Vocab);
            for (float value : logits) if (!std::isfinite(value)) throw std::runtime_error("non-finite logits");
            logits_valid = true;
        }
        ++position;
        if (profile_enabled) {
            const auto total = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - step_started).count());
            const auto covered = (stats.qkv_ns-qkv_before)+(stats.attention_kv_ns-attn_before)+(stats.output_projection_ns-output_before)+(stats.ffn_ns-ffn_before)+(stats.vocab_head_ns-vocab_before);
            stats.remaining_ops_ns += total > covered ? total-covered : 0;
        }
    }
};

static Runtime* checked_runtime(void* runtime) {
    if (runtime == nullptr) throw std::runtime_error("runtime is null");
    return static_cast<Runtime*>(runtime);
}

struct EvalWork {
    Runtime* model = nullptr;
    const int32_t* token_ids = nullptr;
    size_t count = 0;
    float* last_logits = nullptr;
    bool selective = false;
};

static void run_eval_work(void* raw) {
    auto* work = static_cast<EvalWork*>(raw);
    for (size_t i = 0; i < work->count; ++i) work->model->step(work->token_ids[i], !work->selective || i + 1 == work->count);
    if (!work->model->logits_valid) throw std::runtime_error("logits were not produced");
    std::copy(work->model->logits.begin(), work->model->logits.end(), work->last_logits);
}

} // namespace

extern "C" {

MM_RUNTIME_API void* mm_load(const char* path, uint32_t max_context, int kernel_mode,
                             char* error, size_t error_cap) {
    try {
        if (path == nullptr || *path == '\0') throw std::runtime_error("path is empty");
        if (kernel_mode != static_cast<int>(mm::KernelMode::Scalar) && kernel_mode != static_cast<int>(mm::KernelMode::Auto)) {
            throw std::runtime_error("invalid kernel mode");
        }
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open tensor file");
        const std::streamoff size = input.tellg();
        if (size < 0 || static_cast<uint64_t>(size) > kMaxFileBytes) throw std::runtime_error("invalid tensor file size");
        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        input.seekg(0, std::ios::beg);
        if (!bytes.empty() && !input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
            throw std::runtime_error("cannot read tensor file");
        }
        Config config;
        auto tensors = load_tensors(bytes, config);
        if (max_context == 0 || max_context > config.max_positions) throw std::runtime_error("context capacity outside model bounds");
        return new Runtime(config, std::move(tensors), max_context, kernel_mode);
    } catch (const std::exception& ex) {
        set_error(error, error_cap, ex.what());
        return nullptr;
    } catch (...) {
        set_error(error, error_cap, "unknown loader failure");
        return nullptr;
    }
}

MM_RUNTIME_API void mm_free(void* runtime) { delete static_cast<Runtime*>(runtime); }

MM_RUNTIME_API int mm_reset(void* runtime) {
    try {
        Runtime* model = checked_runtime(runtime);
        model->position = 0;
        model->logits_valid = false;
        ++model->cache_epoch;
        return 0;
    } catch (...) {
        return -1;
    }
}

MM_RUNTIME_API uint32_t mm_vocab_size(void* runtime) {
    try { return checked_runtime(runtime)->config.vocab; } catch (...) { return 0; }
}

MM_RUNTIME_API uint32_t mm_position(void* runtime) {
    try { return checked_runtime(runtime)->position; } catch (...) { return 0; }
}

MM_RUNTIME_API const char* mm_backend(void* runtime) {
    static const char* invalid = "invalid";
    try { return checked_runtime(runtime)->backend.c_str(); } catch (...) { return invalid; }
}

MM_RUNTIME_API int mm_configure_profile(void* runtime, int enabled) {
    try {
        Runtime* model = checked_runtime(runtime);
        model->profile_enabled = enabled != 0;
        model->parallel.configure_profile(model->profile_enabled);
        return 0;
    } catch (...) { return -1; }
}

MM_RUNTIME_API int mm_reset_stats(void* runtime) {
    try {
        Runtime* model = checked_runtime(runtime);
        model->stats = {};
        model->parallel.reset_profile_stats();
        return 0;
    } catch (...) { return -1; }
}

MM_RUNTIME_API int mm_get_stats(void* runtime, MmRuntimeStats* out_stats) {
    try {
        Runtime* model = checked_runtime(runtime);
        if (out_stats == nullptr) return -1;
        *out_stats = model->stats;
        const auto parallel = model->parallel.profile_stats();
        for (std::size_t i = 0; i < 64; ++i) {
            out_stats->participant_compute_ns[i] = parallel.participant_compute_ns[i];
            out_stats->participant_compute_calls[i] = parallel.participant_compute_calls[i];
        }
        out_stats->controller_wait_ns = parallel.controller_wait_ns;
        return 0;
    } catch (...) { return -1; }
}

MM_RUNTIME_API uint64_t mm_lm_head_calls(void* runtime) {
    try { return checked_runtime(runtime)->stats.lm_head_calls; } catch (...) { return 0; }
}

MM_RUNTIME_API int mm_logits_valid(void* runtime) { try { return checked_runtime(runtime)->logits_valid ? 1 : 0; } catch (...) { return 0; } }
MM_RUNTIME_API uint64_t mm_cache_epoch(void* runtime) { try { return checked_runtime(runtime)->cache_epoch; } catch (...) { return 0; } }
MM_RUNTIME_API int mm_truncate(void* runtime, uint32_t target, char* error, size_t error_cap) {
    try { Runtime* model = checked_runtime(runtime); if (target > model->position) throw std::runtime_error("cannot advance cache with truncate"); model->position = target; model->logits_valid = false; ++model->cache_epoch; return 0; }
    catch (const std::exception& ex) { set_error(error, error_cap, ex.what()); return -1; } catch (...) { set_error(error, error_cap, "unknown truncate failure"); return -1; }
}

MM_RUNTIME_API int mm_configure_selective_logits(void* runtime, int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    try { checked_runtime(runtime)->selective_logits = enabled != 0; return 0; } catch (...) { return -1; }
}

MM_RUNTIME_API int mm_selective_logits(void* runtime) {
    try { return checked_runtime(runtime)->selective_logits ? 1 : 0; } catch (...) { return 0; }
}

MM_RUNTIME_API int mm_configure_v_blocked_attention(void* runtime, int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    try { checked_runtime(runtime)->v_blocked_attention = enabled != 0; return 0; } catch (...) { return -1; }
}

MM_RUNTIME_API int mm_v_blocked_attention(void* runtime) {
    try { return checked_runtime(runtime)->v_blocked_attention ? 1 : 0; } catch (...) { return 0; }
}

MM_RUNTIME_API int mm_configure_ffn_row4(void* runtime, int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    try { checked_runtime(runtime)->ffn_row4 = enabled != 0; return 0; } catch (...) { return -1; }
}

MM_RUNTIME_API int mm_ffn_row4(void* runtime) {
    try { return checked_runtime(runtime)->ffn_row4 ? 1 : 0; } catch (...) { return 0; }
}

MM_RUNTIME_API int mm_configure_gqa_k_shared(void* runtime, int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    try { checked_runtime(runtime)->gqa_k_shared = enabled != 0; return 0; } catch (...) { return -1; }
}

MM_RUNTIME_API int mm_gqa_k_shared(void* runtime) {
    try { return checked_runtime(runtime)->gqa_k_shared ? 1 : 0; } catch (...) { return 0; }
}

MM_RUNTIME_API int mm_configure_threads(void* runtime, uint32_t threads,
                                         const uint32_t* cpu_indices,
                                         const uint32_t* row_weights,
                                         char* error, size_t error_cap) {
    try {
        checked_runtime(runtime)->parallel.configure(threads, cpu_indices, row_weights);
        return 0;
    } catch (const std::exception& ex) {
        set_error(error, error_cap, ex.what());
        return -1;
    } catch (...) {
        set_error(error, error_cap, "unknown thread configuration failure");
        return -1;
    }
}

MM_RUNTIME_API uint32_t mm_thread_count(void* runtime) {
    try { return checked_runtime(runtime)->parallel.thread_count(); } catch (...) { return 0; }
}

MM_RUNTIME_API int32_t mm_thread_cpu(void* runtime, uint32_t worker) {
    try { return checked_runtime(runtime)->parallel.thread_cpu(worker); } catch (...) { return -2; }
}

MM_RUNTIME_API uint32_t mm_thread_weight(void* runtime, uint32_t worker) {
    try { return checked_runtime(runtime)->parallel.thread_weight(worker); } catch (...) { return 0; }
}

MM_RUNTIME_API int mm_eval(void* runtime, const int32_t* token_ids, size_t count,
                           float* last_logits, size_t logits_capacity, char* error,
                           size_t error_cap) {
    try {
        Runtime* model = checked_runtime(runtime);
        if (token_ids == nullptr || count == 0) throw std::runtime_error("token sequence is empty");
        if (last_logits == nullptr || logits_capacity < model->config.vocab) throw std::runtime_error("logit output capacity is too small");
        if (count > model->max_context - model->position) throw std::runtime_error("context capacity exceeded");
        for (size_t i = 0; i < count; ++i) {
            if (token_ids[i] < 0 || static_cast<uint32_t>(token_ids[i]) >= model->config.vocab) {
                throw std::runtime_error("token ID outside vocabulary");
            }
        }
        const uint32_t old_position = model->position;
        try {
            model->logits_valid = false;
            EvalWork work{model, token_ids, count, last_logits, model->selective_logits};
            model->parallel.execute(run_eval_work, &work);
            if (!model->logits_valid) throw std::runtime_error("logits were not produced");
            ++model->cache_epoch;
        } catch (...) {
            model->position = old_position;
            model->logits_valid = false;
            ++model->cache_epoch;
            throw;
        }
        return 0;
    } catch (const std::exception& ex) {
        set_error(error, error_cap, ex.what());
        return -1;
    } catch (...) {
        set_error(error, error_cap, "unknown evaluation failure");
        return -1;
    }
}

} // extern "C"
