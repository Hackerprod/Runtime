#include "mm_kernels.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sched.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kRows = 6400;
constexpr std::uint32_t kCols = 768;
constexpr std::size_t kLanes = 4;

struct Record {
  std::vector<std::uint16_t> half;
  std::vector<float> weights;
};

bool pin_cpu_zero() {
#if defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1)) != 0;
#else
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
#endif
}

template <typename T>
T read(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("truncated CPU-E7 payload");
  return value;
}

Record load_payload(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open CPU-E7 payload: " + path);
  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "MME7001\0", 8) != 0)
    throw std::runtime_error("invalid CPU-E7 payload magic");
  if (read<std::uint32_t>(in) != 1u) throw std::runtime_error("CPU-E7 payload must contain one record");
  const auto name_length = read<std::uint32_t>(in);
  const auto rows = read<std::uint32_t>(in);
  const auto cols = read<std::uint32_t>(in);
  (void)read<std::uint32_t>(in);
  std::string name(name_length, '\0');
  in.read(name.data(), static_cast<std::streamsize>(name_length));
  if (!in || name != "model.embed_tokens.weight" || rows != kRows || cols != kCols)
    throw std::runtime_error("CPU-E7 payload must contain model.embed_tokens.weight 6400x768");
  Record result;
  result.half.resize(static_cast<std::size_t>(rows) * cols);
  result.weights.resize(result.half.size());
  in.read(reinterpret_cast<char*>(result.half.data()), static_cast<std::streamsize>(result.half.size() * sizeof(std::uint16_t)));
  in.read(reinterpret_cast<char*>(result.weights.data()), static_cast<std::streamsize>(result.weights.size() * sizeof(float)));
  if (!in) throw std::runtime_error("truncated CPU-E7 payload data");
  for (std::size_t i = 0; i < result.weights.size(); ++i)
    if (mm::f32_to_f16(result.weights[i]) != result.half[i])
      throw std::runtime_error("CPU-E7 FP16/FP32 payload mismatch");
  return result;
}

std::uint64_t checksum(const float* values, std::size_t count) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, values + i, sizeof(bits));
    h ^= bits;
    h *= 1099511628211ull;
  }
  return h;
}

struct Result {
  double seconds{};
  std::uint64_t checksum{};
};

void fill_inputs(std::vector<float> (&inputs)[kLanes]) {
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    inputs[lane].resize(kCols);
    for (std::size_t col = 0; col < kCols; ++col)
      inputs[lane][col] = static_cast<float>(((col + 11) * (lane + 3)) % 193) * 0.01f - 0.9f;
  }
}

Result measure_f32(const Record& record, const std::vector<float>& input, float* output, std::size_t reps) {
  const auto start = Clock::now();
  for (std::size_t n = 0; n < reps; ++n)
    mm::gemv_f32(record.weights.data(), input.data(), output, kRows, kCols, mm::KernelMode::Auto);
  return {std::chrono::duration<double>(Clock::now() - start).count(), checksum(output, kRows)};
}

Result measure_f16(const Record& record, const std::vector<float>& input, float* output, std::size_t reps) {
  const auto start = Clock::now();
  for (std::size_t n = 0; n < reps; ++n)
    mm::gemv_f16(record.half.data(), input.data(), output, kRows, kCols, mm::KernelMode::Auto);
  return {std::chrono::duration<double>(Clock::now() - start).count(), checksum(output, kRows)};
}

Result measure_f32_x4(const Record& record, const std::vector<float> (&inputs)[kLanes], float* const outputs[kLanes], std::size_t reps) {
  const auto start = Clock::now();
  for (std::size_t n = 0; n < reps; ++n)
    for (std::size_t lane = 0; lane < kLanes; ++lane)
      mm::gemv_f32(record.weights.data(), inputs[lane].data(), outputs[lane], kRows, kCols, mm::KernelMode::Auto);
  std::uint64_t h = 0;
  for (std::size_t lane = 0; lane < kLanes; ++lane) h ^= checksum(outputs[lane], kRows);
  return {std::chrono::duration<double>(Clock::now() - start).count(), h};
}

Result measure_f16_independent(const Record& record, const std::vector<float> (&inputs)[kLanes], float* const outputs[kLanes], std::size_t reps) {
  const auto start = Clock::now();
  for (std::size_t n = 0; n < reps; ++n)
    for (std::size_t lane = 0; lane < kLanes; ++lane)
      mm::gemv_f16(record.half.data(), inputs[lane].data(), outputs[lane], kRows, kCols, mm::KernelMode::Auto);
  std::uint64_t h = 0;
  for (std::size_t lane = 0; lane < kLanes; ++lane) h ^= checksum(outputs[lane], kRows);
  return {std::chrono::duration<double>(Clock::now() - start).count(), h};
}

Result measure_f16_x4(const Record& record, const std::vector<float> (&inputs)[kLanes], float* const outputs[kLanes], std::size_t reps) {
  const auto start = Clock::now();
  for (std::size_t n = 0; n < reps; ++n)
    mm::gemv_f16_x4(record.half.data(), inputs[0].data(), outputs[0], inputs[1].data(), outputs[1],
                    inputs[2].data(), outputs[2], inputs[3].data(), outputs[3], kRows, kCols,
                    mm::KernelMode::Auto);
  std::uint64_t h = 0;
  for (std::size_t lane = 0; lane < kLanes; ++lane) h ^= checksum(outputs[lane], kRows);
  return {std::chrono::duration<double>(Clock::now() - start).count(), h};
}

void emit(std::size_t pair, const char* order, std::size_t warmups, std::size_t reps,
          const Result& x1a, const Result& x1b, const Result& x4a, const Result& x4b,
          const Result& independent, bool x1_parity, bool x4_parity, bool independent_parity) {
  std::cout << std::fixed << std::setprecision(9)
            << "{\"rows\":6400,\"cols\":768,\"pair\":" << pair
            << ",\"order\":\"" << order << "\",\"repetitions\":" << reps
            << ",\"warmup_repetitions\":" << warmups
            << ",\"x1_f32_seconds\":" << x1a.seconds << ",\"x1_f16_seconds\":" << x1b.seconds
            << ",\"x1_f32_checksum\":" << x1a.checksum << ",\"x1_f16_checksum\":" << x1b.checksum
            << ",\"x1_exact_memcmp_parity\":" << (x1_parity ? "true" : "false")
            << ",\"x4_f32_seconds\":" << x4a.seconds << ",\"x4_f16_x4_seconds\":" << x4b.seconds
            << ",\"x4_f32_checksum\":" << x4a.checksum << ",\"x4_f16_x4_checksum\":" << x4b.checksum
            << ",\"x4_exact_memcmp_parity\":" << (x4_parity ? "true" : "false")
            << ",\"x4_f16_independent_seconds\":" << independent.seconds
            << ",\"x4_f16_independent_checksum\":" << independent.checksum
            << ",\"x4_independent_exact_memcmp_parity\":" << (independent_parity ? "true" : "false")
            << ",\"f16c_available\":" << (mm::f16c_available() ? "true" : "false")
            << ",\"cpu\":0,\"threads\":1}\n";
}

void run(const Record& record, std::size_t warmups, std::size_t reps) {
  std::vector<float> inputs[kLanes];
  fill_inputs(inputs);
  std::vector<float> x1_f32(kRows), x1_f16(kRows), f32_buffers[kLanes], f16_buffers[kLanes], independent_buffers[kLanes];
  float* f32_outputs[kLanes]{};
  float* f16_outputs[kLanes]{};
  float* independent_outputs[kLanes]{};
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    f32_buffers[lane].resize(kRows); f32_outputs[lane] = f32_buffers[lane].data();
    f16_buffers[lane].resize(kRows); f16_outputs[lane] = f16_buffers[lane].data();
    independent_buffers[lane].resize(kRows); independent_outputs[lane] = independent_buffers[lane].data();
  }
  for (std::size_t pair = 1; pair <= 3; ++pair) {
    (void)measure_f32(record, inputs[0], x1_f32.data(), warmups);
    (void)measure_f16(record, inputs[0], x1_f16.data(), warmups);
    (void)measure_f32_x4(record, inputs, f32_outputs, warmups);
    (void)measure_f16_x4(record, inputs, f16_outputs, warmups);
    (void)measure_f16_independent(record, inputs, independent_outputs, warmups);
    const bool ab = pair % 2 == 1;
    Result x1_f32_result{}, x1_f16_result{}, x4_f32_result{}, x4_f16_result{}, independent_result{};
    if (ab) {
      x1_f32_result = measure_f32(record, inputs[0], x1_f32.data(), reps);
      x1_f16_result = measure_f16(record, inputs[0], x1_f16.data(), reps);
      x4_f32_result = measure_f32_x4(record, inputs, f32_outputs, reps);
      x4_f16_result = measure_f16_x4(record, inputs, f16_outputs, reps);
    } else {
      x1_f16_result = measure_f16(record, inputs[0], x1_f16.data(), reps);
      x1_f32_result = measure_f32(record, inputs[0], x1_f32.data(), reps);
      x4_f16_result = measure_f16_x4(record, inputs, f16_outputs, reps);
      x4_f32_result = measure_f32_x4(record, inputs, f32_outputs, reps);
    }
    independent_result = measure_f16_independent(record, inputs, independent_outputs, reps);
    const bool x1_parity = std::memcmp(x1_f32.data(), x1_f16.data(), kRows * sizeof(float)) == 0;
    bool x4_parity = true;
    bool independent_parity = true;
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      x4_parity = x4_parity && std::memcmp(f32_outputs[lane], f16_outputs[lane], kRows * sizeof(float)) == 0;
      independent_parity = independent_parity && std::memcmp(f16_outputs[lane], independent_outputs[lane], kRows * sizeof(float)) == 0;
    }
    if (!x1_parity || !x4_parity || !independent_parity)
      throw std::runtime_error("CPU-E7 exact parity failure");
    emit(pair, ab ? "f32-f16/f32x4-f16x4" : "f16-f32/f16x4-f32x4", warmups, reps,
         x1_f32_result, x1_f16_result, x4_f32_result, x4_f16_result, independent_result,
         x1_parity, x4_parity, independent_parity);
  }
}

std::size_t positive(const char* value, const char* label) {
  try {
    const auto result = std::stoull(value);
    if (!result) throw std::runtime_error("zero");
    return static_cast<std::size_t>(result);
  } catch (...) {
    throw std::runtime_error(std::string("invalid ") + label);
  }
}
}

int main(int argc, char** argv) {
  try {
    std::string payload = "cpu_e7_payload.bin";
    std::size_t warmups = 2, repetitions = 8, pairs = 3;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--payload" && i + 1 < argc) payload = argv[++i];
      else if (argument == "--warmups" && i + 1 < argc) warmups = positive(argv[++i], "warmups");
      else if (argument == "--repetitions" && i + 1 < argc) repetitions = positive(argv[++i], "repetitions");
      else if (argument == "--pairs" && i + 1 < argc) pairs = positive(argv[++i], "pairs");
      else throw std::runtime_error("usage: mm_bench_cpu_e7 [--payload FILE] [--pairs 3] [--warmups N] [--repetitions 8]");
    }
    if (pairs != 3) throw std::runtime_error("CPU-E7 requires exactly three alternating pairs");
    if (!pin_cpu_zero()) throw std::runtime_error("failed to pin benchmark participant to CPU 0");
    run(load_payload(payload), warmups, repetitions);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
