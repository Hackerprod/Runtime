#include "mm_runtime.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

void u32(std::vector<uint8_t>& out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void u64(std::vector<uint8_t>& out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void f32(std::vector<uint8_t>& out, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(out, bits);
}
uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

void tensor(std::vector<uint8_t>& out, const std::string& name,
            const std::vector<uint32_t>& shape, float value = 0.0f,
            bool q4 = false, float q4_scale = 1.0f) {
    u32(out, static_cast<uint32_t>(name.size()));
    out.insert(out.end(), name.begin(), name.end());
    u32(out, q4 ? 1u : 0u);
    u32(out, static_cast<uint32_t>(shape.size()));
    size_t elements = 1;
    for (uint32_t dim : shape) { u32(out, dim); elements *= dim; }
    if (!q4) {
        u32(out, 0); // group size
        u64(out, elements * sizeof(float));
        u64(out, 0); // scale count
        for (size_t i = 0; i < elements; ++i) f32(out, value);
        return;
    }
    const size_t rows = shape[0], cols = shape[1];
    const size_t row_bytes = (cols + 1) / 2;
    const size_t groups = (cols + 31) / 32;
    u32(out, 32); // group size
    u64(out, rows * row_bytes);
    u64(out, rows * groups);
    out.insert(out.end(), rows * row_bytes, 0x88); // every decoded value is zero
    for (size_t i = 0; i < rows * groups; ++i) f32(out, q4_scale);
}

std::vector<uint8_t> valid_fixture(bool q4_proj = false, float q4_scale = 1.0f,
                                   bool execution_overflow = false) {
    constexpr uint32_t vocab = 8, hidden = 4, layers = 1, q = 2, kv = 1, hd = 2, inter = 4;
    std::vector<std::pair<std::string, std::vector<uint32_t>>> records = {
        {"model.embed_tokens.weight", {vocab, hidden}},
        {"model.norm.weight", {hidden}},
        {"model.layers.0.input_layernorm.weight", {hidden}},
        {"model.layers.0.post_attention_layernorm.weight", {hidden}},
        {"model.layers.0.self_attn.q_proj.weight", {q * hd, hidden}},
        {"model.layers.0.self_attn.k_proj.weight", {kv * hd, hidden}},
        {"model.layers.0.self_attn.v_proj.weight", {kv * hd, hidden}},
        {"model.layers.0.self_attn.o_proj.weight", {hidden, q * hd}},
        {"model.layers.0.self_attn.q_norm.weight", {hd}},
        {"model.layers.0.self_attn.k_norm.weight", {hd}},
        {"model.layers.0.mlp.gate_proj.weight", {inter, hidden}},
        {"model.layers.0.mlp.up_proj.weight", {inter, hidden}},
        {"model.layers.0.mlp.down_proj.weight", {hidden, inter}},
    };
    std::vector<uint8_t> out;
    out.insert(out.end(), {'M','M','C','P','U','0','0','1'});
    u32(out, 1); u32(out, vocab); u32(out, hidden); u32(out, layers);
    u32(out, q); u32(out, kv); u32(out, hd); u32(out, inter);
    u32(out, 4); u32(out, 2); u32(out, 1); u32(out, static_cast<uint32_t>(records.size()));
    f32(out, 1e-6f); f32(out, 1e6f);
    for (const auto& record : records) {
        const bool use_q4 = q4_proj && record.first == "model.layers.0.self_attn.q_proj.weight";
        const bool embedding = record.first == "model.embed_tokens.weight";
        const float value = !execution_overflow ? 0.0f
            : embedding ? 1.0f : record.first == "model.norm.weight" ? 3e38f : 0.0f;
        tensor(out, record.first, record.second, value, use_q4, q4_scale);
        if (execution_overflow && embedding) {
            // Token 0 stays finite; other tokens overflow only at the tied head.
            const auto payload = out.end() - static_cast<std::vector<uint8_t>::difference_type>(
                vocab * hidden * sizeof(float));
            std::fill_n(payload, hidden * sizeof(float), uint8_t{0});
        }
    }
    u32(out, crc32(out.data(), out.size()));
    return out;
}

void patch_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    assert(offset + 4 <= bytes.size());
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}

void patch_crc(std::vector<uint8_t>& bytes) {
    assert(bytes.size() >= 4);
    patch_u32(bytes, bytes.size() - 4, crc32(bytes.data(), bytes.size() - 4));
}

void replace_name(std::vector<uint8_t>& bytes, const std::string& from, const std::string& to) {
    assert(from.size() == to.size());
    const auto begin = bytes.begin();
    const auto it = std::search(begin, bytes.end(), from.begin(), from.end());
    assert(it != bytes.end());
    std::copy(to.begin(), to.end(), it);
}

std::filesystem::path write_file(const std::vector<uint8_t>& bytes, const char* suffix) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        (std::string("mmcpu_runtime_") + std::to_string(stamp) + suffix);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(file.good());
    return path;
}

void test_cache_reset_and_overflow() {
    const auto path = write_file(valid_fixture(), ".bin");
    char error[256]{};
    void* runtime = mm_load(path.string().c_str(), 3, 0, error, sizeof(error));
    assert(runtime != nullptr);
    assert(mm_vocab_size(runtime) == 8);
    assert(mm_position(runtime) == 0);
    assert(std::string(mm_backend(runtime)) == "scalar");
    float logits[8]{};
    const int32_t prompt[] = {1, 2};
    assert(mm_eval(runtime, prompt, 2, logits, 8, error, sizeof(error)) == 0);
    assert(mm_position(runtime) == 2);
    const int32_t too_long[] = {3, 4};
    assert(mm_eval(runtime, too_long, 2, logits, 8, error, sizeof(error)) != 0);
    assert(mm_position(runtime) == 2);
    assert(mm_reset(runtime) == 0);
    assert(mm_position(runtime) == 0);
    const int32_t invalid[] = {8};
    assert(mm_eval(runtime, invalid, 1, logits, 8, error, sizeof(error)) != 0);
    assert(mm_position(runtime) == 0);
    mm_free(runtime);
    std::filesystem::remove(path);
}

void test_crc_and_truncation() {
    auto bytes = valid_fixture();
    bytes.back() ^= 0x01u;
    const auto bad_crc = write_file(bytes, ".badcrc");
    char error[256]{};
    assert(mm_load(bad_crc.string().c_str(), 3, 0, error, sizeof(error)) == nullptr);
    assert(std::string(error).find("CRC32") != std::string::npos);
    std::filesystem::remove(bad_crc);

    bytes = valid_fixture();
    bytes.pop_back();
    const auto truncated = write_file(bytes, ".truncated");
    std::memset(error, 0, sizeof(error));
    assert(mm_load(truncated.string().c_str(), 3, 0, error, sizeof(error)) == nullptr);
    assert(error[0] != '\0');
    std::filesystem::remove(truncated);
}

void test_rejects_malformed_metadata() {
    char error[256]{};
    auto bytes = valid_fixture();
    patch_u32(bytes, 36, 5); // intermediate size disagrees with tensor descriptors
    patch_crc(bytes);
    auto path = write_file(bytes, ".shape");
    assert(mm_load(path.string().c_str(), 3, 0, error, sizeof(error)) == nullptr);
    std::filesystem::remove(path);

    bytes = valid_fixture();
    replace_name(bytes, "model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight");
    patch_crc(bytes);
    path = write_file(bytes, ".duplicate");
    assert(mm_load(path.string().c_str(), 3, 0, error, sizeof(error)) == nullptr);
    std::filesystem::remove(path);

    bytes = valid_fixture();
    patch_u32(bytes, 56, 0x7fc00000u); // NaN RMS epsilon
    patch_crc(bytes);
    path = write_file(bytes, ".nonfinite");
    assert(mm_load(path.string().c_str(), 3, 0, error, sizeof(error)) == nullptr);
    std::filesystem::remove(path);
}

void test_q4_validation_and_dispatch() {
    char error[256]{};
    auto bytes = valid_fixture(true, 1.0f);
    auto path = write_file(bytes, ".q4");
    void* runtime = mm_load(path.string().c_str(), 3, 0, error, sizeof(error));
    assert(runtime != nullptr);
    float logits[8]{};
    const int32_t token = 1;
    assert(mm_eval(runtime, &token, 1, logits, 8, error, sizeof(error)) == 0);
    mm_free(runtime);
    std::filesystem::remove(path);

    bytes = valid_fixture(true, 0.0f);
    path = write_file(bytes, ".q4zero");
    assert(mm_load(path.string().c_str(), 3, 0, error, sizeof(error)) == nullptr);
    std::filesystem::remove(path);
}

void test_runtime_table_safety_bound() {
    char error[256]{};
    auto bytes = valid_fixture();
    patch_u32(bytes, 40, 100'000'000u); // valid header, but absurd requested capacity
    patch_crc(bytes);
    auto path = write_file(bytes, ".huge");
    assert(mm_load(path.string().c_str(), 100'000'000u, 0, error, sizeof(error)) == nullptr);
    assert(std::string(error).find("safety") != std::string::npos);
    std::filesystem::remove(path);
}

// Nonzero, two-layer fixtures expose skipped/overlapping row shards and stale
// KV state. Odd FFN/output widths and Q4 group tails prevent aligned-only tests.
constexpr uint32_t thread_vocab = 43;

void patterned_tensor(std::vector<uint8_t>& out, const std::string& name,
                      const std::vector<uint32_t>& shape, uint32_t salt, bool q4) {
    u32(out, static_cast<uint32_t>(name.size()));
    out.insert(out.end(), name.begin(), name.end());
    u32(out, q4 ? 1u : 0u);
    u32(out, static_cast<uint32_t>(shape.size()));
    size_t elements = 1;
    for (uint32_t dim : shape) { u32(out, dim); elements *= dim; }
    const auto signed_value = [salt](size_t index) {
        return static_cast<int>((index * 7 + salt * 11) % 15) - 7;
    };
    if (!q4) {
        u32(out, 0); u64(out, elements * sizeof(float)); u64(out, 0);
        for (size_t i = 0; i < elements; ++i) {
            const float value = shape.size() == 1
                ? 1.0f + static_cast<float>(i % 3) / 64.0f
                : static_cast<float>(signed_value(i)) / 32.0f;
            f32(out, value);
        }
        return;
    }
    const size_t rows = shape[0], cols = shape[1];
    const size_t row_bytes = (cols + 1) / 2, groups = (cols + 31) / 32;
    u32(out, 32); u64(out, rows * row_bytes); u64(out, rows * groups);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; col += 2) {
            const auto low = static_cast<uint8_t>(signed_value(row * cols + col) + 8);
            const auto high = static_cast<uint8_t>(col + 1 < cols
                ? signed_value(row * cols + col + 1) + 8 : 8);
            out.push_back(static_cast<uint8_t>(low | (high << 4)));
        }
    }
    for (size_t i = 0; i < rows * groups; ++i) f32(out, 1.0f / 32.0f);
}

std::vector<uint8_t> threading_fixture(bool q4) {
    constexpr uint32_t hidden = 36, layers = 2, q = 3, kv = 1, hd = 12, inter = 41;
    std::vector<std::pair<std::string, std::vector<uint32_t>>> records = {
        {"model.embed_tokens.weight", {thread_vocab, hidden}},
        {"model.norm.weight", {hidden}},
    };
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        const std::vector<std::pair<std::string, std::vector<uint32_t>>> local = {
            {"input_layernorm.weight", {hidden}}, {"post_attention_layernorm.weight", {hidden}},
            {"self_attn.q_proj.weight", {q * hd, hidden}},
            {"self_attn.k_proj.weight", {kv * hd, hidden}},
            {"self_attn.v_proj.weight", {kv * hd, hidden}},
            {"self_attn.o_proj.weight", {hidden, q * hd}},
            {"self_attn.q_norm.weight", {hd}}, {"self_attn.k_norm.weight", {hd}},
            {"mlp.gate_proj.weight", {inter, hidden}}, {"mlp.up_proj.weight", {inter, hidden}},
            {"mlp.down_proj.weight", {hidden, inter}},
        };
        for (const auto& record : local) records.push_back({prefix + record.first, record.second});
    }
    std::vector<uint8_t> out{'M','M','C','P','U','0','0','1'};
    u32(out, 1); u32(out, thread_vocab); u32(out, hidden); u32(out, layers);
    u32(out, q); u32(out, kv); u32(out, hd); u32(out, inter);
    u32(out, 16); u32(out, 2); u32(out, 1); u32(out, static_cast<uint32_t>(records.size()));
    f32(out, 1e-6f); f32(out, 1e6f);
    uint32_t salt = 1;
    for (const auto& record : records) {
        const bool quantized = q4 && record.second.size() == 2
            && record.first != "model.embed_tokens.weight";
        patterned_tensor(out, record.first, record.second, salt++, quantized);
    }
    u32(out, crc32(out.data(), out.size()));
    return out;
}

struct ScopedRuntime {
    void* value;
    ScopedRuntime(const std::filesystem::path& path, int mode) {
        char error[256]{};
        value = mm_load(path.string().c_str(), 16, mode, error, sizeof(error));
        if (!value) std::cerr << error << '\n';
        assert(value != nullptr);
    }
    ~ScopedRuntime() { mm_free(value); }
    ScopedRuntime(const ScopedRuntime&) = delete;
    ScopedRuntime& operator=(const ScopedRuntime&) = delete;
};

using Logits = std::array<float, thread_vocab>;

Logits evaluate(void* runtime, const std::vector<int32_t>& ids) {
    char error[256]{};
    Logits result{};
    const int status = mm_eval(runtime, ids.data(), ids.size(), result.data(),
                               result.size(), error, sizeof(error));
    if (status != 0) std::cerr << error << '\n';
    assert(status == 0);
    for (float value : result) assert(std::isfinite(value));
    return result;
}

void assert_exact(const Logits& actual, const Logits& expected) {
    assert(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0);
}

void configure(void* runtime, uint32_t threads, const uint32_t* weights = nullptr,
               const uint32_t* cpus = nullptr) {
    char error[256]{};
    const uint32_t position = mm_position(runtime);
    const int status = mm_configure_threads(runtime, threads, cpus, weights, error, sizeof(error));
    if (status != 0) std::cerr << error << '\n';
    assert(status == 0);
    assert(mm_position(runtime) == position);
    assert(mm_thread_count(runtime) == threads);
    for (uint32_t i = 0; i < threads; ++i) {
        assert(mm_thread_cpu(runtime, i) == (cpus ? static_cast<int32_t>(cpus[i]) : -1));
        assert(mm_thread_weight(runtime, i) == (weights ? weights[i] : 1u));
    }
    assert(mm_thread_cpu(runtime, threads) == -2);
}

void test_threaded_nonzero_parity_and_reconfiguration() {
    for (bool q4 : {false, true}) {
        const auto path = write_file(threading_fixture(q4), q4 ? ".thread-q4" : ".thread-fp32");
        for (int mode : {0, 1}) {
            ScopedRuntime serial(path, mode), candidate(path, mode);
            assert(mm_thread_count(candidate.value) == 1);
            assert(mm_thread_cpu(candidate.value, 0) == -1);
            assert(mm_thread_cpu(candidate.value, 1) == -2);
            assert(mm_thread_weight(candidate.value, 0) == 1);
            const auto initial = evaluate(serial.value, {1, 4});
            assert(std::any_of(initial.begin(), initial.end(), [](float x) { return x != 0.0f; }));
            assert(std::adjacent_find(initial.begin(), initial.end(), std::not_equal_to<float>()) != initial.end());
            assert_exact(evaluate(candidate.value, {1, 4}), initial);

            const uint32_t two_weights[] = {2, 5};
            configure(candidate.value, 2, two_weights);
            assert_exact(evaluate(candidate.value, {7}), evaluate(serial.value, {7}));
            const uint32_t four_weights[] = {1, 3, 5, 8};
            configure(candidate.value, 4, four_weights);
            assert_exact(evaluate(candidate.value, {2, 9}), evaluate(serial.value, {2, 9}));
            configure(candidate.value, 1);
            assert_exact(evaluate(candidate.value, {3}), evaluate(serial.value, {3}));

            for (uint32_t threads : {2u, 4u}) {
                configure(candidate.value, threads);
                for (int repetition = 0; repetition < 3; ++repetition) {
                    assert(mm_reset(serial.value) == 0);
                    assert(mm_reset(candidate.value) == 0);
                    assert_exact(evaluate(candidate.value, {1, 4, 7, 2}), evaluate(serial.value, {1, 4, 7, 2}));
                    assert_exact(evaluate(candidate.value, {9}), evaluate(serial.value, {9}));
                }
            }
            // Valid positive weights can assign empty row spans to workers.
            const uint32_t empty_spans[] = {1, 1, 1, UINT32_MAX};
            configure(candidate.value, 4, empty_spans);
            assert_exact(evaluate(candidate.value, {11}), evaluate(serial.value, {11}));
            configure(candidate.value, 64); // Inclusive configuration upper bound.
            configure(candidate.value, 1);  // Tear down an idle large team safely.
        }
        std::filesystem::remove(path);
    }
}

void test_invalid_thread_config_is_transactional() {
    const auto path = write_file(threading_fixture(false), ".thread-invalid");
    {
        ScopedRuntime serial(path, 0), candidate(path, 0);
        const uint32_t weights[] = {2, 5};
        configure(candidate.value, 2, weights);
        assert_exact(evaluate(candidate.value, {1, 4}), evaluate(serial.value, {1, 4}));
        const auto reject = [&](uint32_t count, const uint32_t* cpus, const uint32_t* requested_weights) {
            char error[256]{};
            const auto position = mm_position(candidate.value);
            assert(mm_configure_threads(candidate.value, count, cpus, requested_weights, error, sizeof(error)) != 0);
            assert(error[0] != '\0');
            assert(mm_position(candidate.value) == position);
            assert(mm_thread_count(candidate.value) == 2);
            for (uint32_t i = 0; i < 2; ++i) {
                assert(mm_thread_cpu(candidate.value, i) == -1);
                assert(mm_thread_weight(candidate.value, i) == weights[i]);
            }
            assert_exact(evaluate(candidate.value, {3}), evaluate(serial.value, {3}));
        };
        reject(0, nullptr, nullptr);
        reject(65, nullptr, nullptr);
        const uint32_t zero_weights[] = {1, 0};
        reject(2, nullptr, zero_weights);
        const uint32_t invalid_cpu[] = {UINT32_MAX};
        reject(1, invalid_cpu, nullptr);
        const uint32_t duplicate_cpus[] = {0, 0};
        reject(2, duplicate_cpus, nullptr);
        char tiny_error[1] = {'x'};
        assert(mm_configure_threads(candidate.value, 0, nullptr, nullptr, tiny_error, 1) != 0);
        assert(tiny_error[0] == '\0');
        assert(mm_thread_count(candidate.value) == 2);
        configure(candidate.value, 4);
        assert_exact(evaluate(candidate.value, {5}), evaluate(serial.value, {5}));
    } // Destruction joins a configured parallel team; no dangling-handle calls.
    std::filesystem::remove(path);
}

void test_new_team_immediate_dispatch() {
    const auto path = write_file(threading_fixture(false), ".thread-startup");
    {
        ScopedRuntime serial(path, 0), candidate(path, 0);
        const auto expected = evaluate(serial.value, {1, 4});
        const int32_t ids[] = {1, 4};
        char error[256]{};
        // No metadata queries between configuration and the first job: a late
        // worker must not initialize its generation counter past that job.
        // Run under CTest's external timeout so a missed wakeup fails boundedly.
        for (unsigned cycle = 0; cycle < 32; ++cycle) {
            Logits actual{};
            const uint32_t threads = cycle % 2 == 0 ? 2u : 4u;
            assert(mm_configure_threads(candidate.value, threads, nullptr, nullptr, error, sizeof(error)) == 0);
            assert(mm_eval(candidate.value, ids, 2, actual.data(), actual.size(), error, sizeof(error)) == 0);
            assert_exact(actual, expected);
            assert(mm_reset(candidate.value) == 0);
        }
        assert_exact(evaluate(candidate.value, {1, 4}), expected);
        // A parked team must wake for the next token without losing KV state.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        assert_exact(evaluate(candidate.value, {7}), evaluate(serial.value, {7}));
    }
    std::filesystem::remove(path);
}

#if defined(_WIN32)
void test_pinned_workers_preserve_caller_affinity() {
    GROUP_AFFINITY before{};
    assert(GetThreadGroupAffinity(GetCurrentThread(), &before));
    // The ABI exposes group-0-local CPU indices, not flattened global indices.
    if (before.Group != 0) return;
    std::vector<uint32_t> cpus;
    for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8 && cpus.size() < 2; ++bit) {
        if (before.Mask & (static_cast<KAFFINITY>(1) << bit)) cpus.push_back(bit);
    }
    if (cpus.size() < 2) return; // No assumption about available laptop CPUs.
    const auto assert_affinity_restored = [&before]() {
        GROUP_AFFINITY current{};
        assert(GetThreadGroupAffinity(GetCurrentThread(), &current));
        assert(current.Group == before.Group && current.Mask == before.Mask);
    };
    const auto assert_invalid_eval_restores_affinity = [&](void* handle) {
        const auto position = mm_position(handle);
        const int32_t invalid_token = static_cast<int32_t>(thread_vocab);
        Logits logits{};
        char error[256]{};
        assert(mm_eval(handle, &invalid_token, 1, logits.data(), thread_vocab,
                       error, sizeof(error)) != 0);
        assert(error[0] != '\0');
        assert(mm_position(handle) == position);
        assert_affinity_restored();
    };
    const auto path = write_file(threading_fixture(false), ".thread-pinned");
    {
        ScopedRuntime serial(path, 0), candidate(path, 0);
        const uint32_t weights[] = {1, 3};
        configure(candidate.value, 2, weights, cpus.data());
        assert_affinity_restored();
        assert_exact(evaluate(candidate.value, {1, 4, 7}), evaluate(serial.value, {1, 4, 7}));
        assert_affinity_restored();
        assert_invalid_eval_restores_affinity(candidate.value);
        // An explicitly pinned one-thread team must restore caller affinity too.
        configure(candidate.value, 1, nullptr, cpus.data());
        assert_affinity_restored();
        assert_exact(evaluate(candidate.value, {2}), evaluate(serial.value, {2}));
        assert_affinity_restored();
        assert_invalid_eval_restores_affinity(candidate.value);
    }
    assert_affinity_restored();
    std::filesystem::remove(path);

    const auto overflow_path = write_file(valid_fixture(false, 1.0f, true), ".thread-overflow");
    for (const uint32_t threads : {1u, 2u}) {
        char error[256]{};
        void* runtime = mm_load(overflow_path.string().c_str(), 3, 0, error, sizeof(error));
        assert(runtime != nullptr); // All serialized FP32 values are finite.
        configure(runtime, threads, nullptr, cpus.data());
        assert_affinity_restored();
        float logits[8]{};
        const int32_t zero = 0;
        assert(mm_eval(runtime, &zero, 1, logits, 8, error, sizeof(error)) == 0);
        assert(mm_position(runtime) == 1);
        assert_affinity_restored();
        const int32_t partial_batch[] = {0, 1};
        assert(mm_eval(runtime, partial_batch, 2, logits, 8, error, sizeof(error)) != 0);
        assert(std::string(error).find("non-finite logits") != std::string::npos);
        assert(mm_position(runtime) == 1); // Roll back the first successful batch token.
        assert_affinity_restored();
        assert(mm_eval(runtime, &zero, 1, logits, 8, error, sizeof(error)) == 0);
        assert(mm_position(runtime) == 2);
        for (const float value : logits) assert(value == 0.0f);
        assert_affinity_restored();
        mm_free(runtime);
        assert_affinity_restored();
    }
    std::filesystem::remove(overflow_path);
}
#endif

} // namespace

void test_diagnostic_profile_contract() {
    const auto path = write_file(threading_fixture(false), ".profile");
    for (int mode : {0, 1}) {
        ScopedRuntime reference(path, mode);
        const auto expected = evaluate(reference.value, {1, 4});
        for (uint32_t threads : {1u, 2u, 4u}) {
            ScopedRuntime candidate(path, mode);
            // Profiling must survive creating/replacing the persistent team.
            assert(mm_configure_profile(candidate.value, 1) == 0);
            configure(candidate.value, threads);
            assert_exact(evaluate(candidate.value, {1, 4}), expected);
            MmRuntimeStats stats{};
            assert(mm_get_stats(candidate.value, &stats) == 0);
            assert(stats.lm_head_calls == 2 && mm_lm_head_calls(candidate.value) == 2);
            assert(stats.qkv_calls == 12 && stats.attention_kv_calls == 4);
            assert(stats.output_projection_calls == 4 && stats.ffn_calls == 12);
            assert(stats.vocab_head_calls == 2);
            assert(stats.qkv_ns > 0 && stats.attention_kv_ns > 0);
            assert(stats.remaining_ops_ns > 0 && stats.vocab_head_ns > 0);
            for (uint32_t i = 0; i < 64; ++i) {
                assert(stats.participant_compute_calls[i] == (i < threads ? 30u : 0u));
                assert((stats.participant_compute_ns[i] > 0) == (i < threads));
            }
            assert(mm_position(candidate.value) == 2);
            assert(mm_reset_stats(candidate.value) == 0);
            assert(mm_position(candidate.value) == 2);
            assert(mm_lm_head_calls(candidate.value) == 0);
            assert_exact(evaluate(candidate.value, {3}), evaluate(reference.value, {3}));
            // Model reset never resets the independently scoped statistics.
            assert(mm_reset(candidate.value) == 0);
            assert(mm_position(candidate.value) == 0);
            assert(mm_lm_head_calls(candidate.value) == 1);
            assert(mm_reset_stats(candidate.value) == 0);
            assert(mm_configure_profile(candidate.value, 0) == 0);
            assert_exact(evaluate(candidate.value, {1, 4}), expected);
            assert(mm_get_stats(candidate.value, &stats) == 0);
            assert(stats.lm_head_calls == 2);
            assert(stats.qkv_ns == 0 && stats.remaining_ops_ns == 0);
            assert(stats.controller_wait_ns == 0);
            for (uint32_t i = 0; i < 64; ++i) assert(stats.participant_compute_ns[i] == 0);
            // Restore the reference prefix for the next participant count.
            assert(mm_reset(reference.value) == 0);
            assert_exact(evaluate(reference.value, {1, 4}), expected);
        }
    }
    assert(mm_configure_profile(nullptr, 1) == -1);
    assert(mm_reset_stats(nullptr) == -1);
    assert(mm_get_stats(nullptr, nullptr) == -1);
    std::filesystem::remove(path);
}

void test_selective_logits_contract() {
    const auto path = write_file(threading_fixture(false), ".selective");
    const std::vector<int32_t> ids = {1, 4, 7, 2, 9, 11, 3, 8};
    for (int mode : {0, 1}) {
        // Establish an all-head reference at every logical position.  The
        // candidate is allowed to return only the endpoint of each chunk.
        char error[256]{};
        ScopedRuntime reference(path, mode);
        std::vector<Logits> expected;
        for (const int32_t token : ids) expected.push_back(evaluate(reference.value, {token}));
        assert(mm_position(reference.value) == ids.size());
        assert(mm_lm_head_calls(reference.value) == ids.size());

        for (uint32_t threads : {1u, 2u, 4u}) {
            for (const std::vector<size_t>& partitions : {
                std::vector<size_t>{8}, std::vector<size_t>{1,1,1,1,1,1,1,1},
                std::vector<size_t>{2,2,2,2}, std::vector<size_t>{3,3,2},
                std::vector<size_t>{4,4}}) {
                for (const std::vector<int32_t>& prefix : {
                    std::vector<int32_t>{}, std::vector<int32_t>{1,4}}) {
                    std::vector<Logits> prefix_expected;
                    if (prefix.empty()) {
                        prefix_expected = expected;
                    } else {
                        ScopedRuntime prefix_reference(path, mode);
                        (void)evaluate(prefix_reference.value, prefix);
                        for (const int32_t token : ids)
                            prefix_expected.push_back(evaluate(prefix_reference.value, {token}));
                    }
                    ScopedRuntime candidate(path, mode);
                    configure(candidate.value, threads);
                    assert(mm_configure_selective_logits(candidate.value, 1) == 0);
                    assert(mm_selective_logits(candidate.value) == 1);
                    assert(mm_reset(candidate.value) == 0);
                    if (!prefix.empty()) {
                        (void)evaluate(candidate.value, prefix);
                        assert(mm_position(candidate.value) == prefix.size());
                    }
                    assert(mm_reset_stats(candidate.value) == 0);
                    size_t offset = 0;
                    uint64_t calls = 0;
                    for (const size_t width : partitions) {
                        assert(offset + width <= ids.size());
                        std::vector<int32_t> chunk(ids.begin() + offset, ids.begin() + offset + width);
                        const Logits actual = evaluate(candidate.value, chunk);
                        assert_exact(actual, prefix_expected[offset + width - 1]);
                        offset += width;
                        ++calls;
                        assert(mm_position(candidate.value) == prefix.size() + offset);
                    }
                    assert(offset == ids.size());
                    assert(mm_lm_head_calls(candidate.value) == calls);
                    assert(mm_configure_selective_logits(candidate.value, 0) == 0);
                    assert(mm_selective_logits(candidate.value) == 0);
                }
            }
        }

        // Toggling the optimization off after a cached prefix must preserve
        // the continuation exactly, not invalidate or replay the KV cache.
        ScopedRuntime toggled(path, mode), control(path, mode);
        configure(toggled.value, 2);
        configure(control.value, 2);
        assert(mm_configure_selective_logits(toggled.value, 1) == 0);
        (void)evaluate(toggled.value, {ids[0], ids[1]});
        (void)evaluate(control.value, {ids[0], ids[1]});
        assert(mm_configure_selective_logits(toggled.value, 0) == 0);
        assert_exact(evaluate(toggled.value, {ids[2], ids[3]}),
                     evaluate(control.value, {ids[2], ids[3]}));

        // Invalid input and context overflow are transactional in selective mode.
        assert(mm_reset(toggled.value) == 0);
        assert(mm_configure_selective_logits(toggled.value, 1) == 0);
        (void)evaluate(toggled.value, {ids[0], ids[1]});
        const uint32_t before = mm_position(toggled.value);
        const int32_t invalid = static_cast<int32_t>(thread_vocab);
        Logits output{};
        assert(mm_eval(toggled.value, &invalid, 1, output.data(), output.size(), error, sizeof(error)) != 0);
        assert(mm_position(toggled.value) == before && error[0] != '\0');
        std::vector<int32_t> too_long(15, ids[0]);
        assert(mm_eval(toggled.value, too_long.data(), too_long.size(), output.data(), output.size(), error, sizeof(error)) != 0);
        assert(mm_position(toggled.value) == before);
    }
    std::filesystem::remove(path);
}

int main() {
    test_diagnostic_profile_contract();
    test_selective_logits_contract();
    test_cache_reset_and_overflow();
    test_crc_and_truncation();
    test_rejects_malformed_metadata();
    test_q4_validation_and_dispatch();
    test_runtime_table_safety_bound();
    test_threaded_nonzero_parity_and_reconfiguration();
    test_invalid_thread_config_is_transactional();
    test_new_team_immediate_dispatch();
#if defined(_WIN32)
    test_pinned_workers_preserve_caller_affinity();
#endif
    std::cout << "mm runtime tests passed\n";
    return 0;
}
