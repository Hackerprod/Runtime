#include "mm_kernels.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

bool pin_cpu_zero() {
#if defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1)) != 0;
#else
  return true;
#endif
}

std::uint64_t checksum(const std::vector<float>& values) {
  std::uint64_t hash = 1469598103934665603ull;
  for (float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hash ^= bits;
    hash *= 1099511628211ull;
  }
  return hash;
}

struct TimedResult {
  double seconds = 0.0;
  std::uint64_t checksum = 0;
};

TimedResult measure_reference(const std::vector<std::uint16_t>& weights,
                              const std::vector<float>* inputs,
                              std::vector<float>* outputs,
                              std::size_t rows, std::size_t cols,
                              std::size_t repetitions) {
  const auto started = Clock::now();
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      mm::gemv_f16(weights.data(), inputs[lane].data(), outputs[lane].data(),
                   rows, cols, mm::KernelMode::Auto);
    }
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
  std::uint64_t hash = 0;
  for (std::size_t lane = 0; lane < 4; ++lane) hash ^= checksum(outputs[lane]);
  return {seconds, hash};
}

TimedResult measure_shared(const std::vector<std::uint16_t>& weights,
                           const std::vector<float>* inputs,
                           std::vector<float>* outputs,
                           std::size_t rows, std::size_t cols,
                           std::size_t repetitions) {
  const auto started = Clock::now();
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    mm::gemv_f16_x4(weights.data(), inputs[0].data(), outputs[0].data(),
                    inputs[1].data(), outputs[1].data(), inputs[2].data(), outputs[2].data(),
                    inputs[3].data(), outputs[3].data(), rows, cols,
                    mm::KernelMode::Auto);
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
  std::uint64_t hash = 0;
  for (std::size_t lane = 0; lane < 4; ++lane) hash ^= checksum(outputs[lane]);
  return {seconds, hash};
}

void warmup(const std::vector<std::uint16_t>& weights,
            const std::vector<float>* inputs, std::vector<float>* outputs,
            std::size_t rows, std::size_t cols, std::size_t repetitions) {
  (void)measure_reference(weights, inputs, outputs, rows, cols, repetitions);
  (void)measure_shared(weights, inputs, outputs, rows, cols, repetitions);
}

void run_shape(const char* shape, std::size_t rows, std::size_t cols,
               std::size_t pairs, std::size_t repetitions) {
  std::vector<std::uint16_t> weights(rows * cols);
  std::vector<float> inputs[4];
  std::vector<float> reference[4];
  std::vector<float> shared[4];
  for (auto& input : inputs) input.resize(cols);
  for (auto& output : reference) output.resize(rows);
  for (auto& output : shared) output.resize(rows);

  for (std::size_t index = 0; index < weights.size(); ++index) {
    const float value = static_cast<float>(static_cast<int>((index * 37u) % 61u) - 30) * 0.03125f;
    weights[index] = mm::f32_to_f16(value);
  }
  for (std::size_t lane = 0; lane < 4; ++lane) {
    for (std::size_t col = 0; col < cols; ++col) {
      const float phase = static_cast<float>(col + 1u) * (0.071f + 0.013f * static_cast<float>(lane));
      inputs[lane][col] = std::sin(phase) + 0.017f * static_cast<float>(lane + 1u);
    }
  }

  for (std::size_t lane = 0; lane < 4; ++lane) {
    mm::gemv_f16(weights.data(), inputs[lane].data(), reference[lane].data(), rows, cols);
  }
  mm::gemv_f16_x4(weights.data(), inputs[0].data(), shared[0].data(),
                  inputs[1].data(), shared[1].data(), inputs[2].data(), shared[2].data(),
                  inputs[3].data(), shared[3].data(), rows, cols);
  for (std::size_t lane = 0; lane < 4; ++lane) {
    if (std::memcmp(reference[lane].data(), shared[lane].data(), rows * sizeof(float)) != 0) {
      throw std::runtime_error("CPU-E1 parity failure before timing");
    }
  }

  for (std::size_t pair = 1; pair <= pairs; ++pair) {
    warmup(weights, inputs, reference, rows, cols, 2);
    warmup(weights, inputs, shared, rows, cols, 2);
    TimedResult reference_result;
    TimedResult shared_result;
    if (pair % 2u == 1u) {
      reference_result = measure_reference(weights, inputs, reference, rows, cols, repetitions);
      shared_result = measure_shared(weights, inputs, shared, rows, cols, repetitions);
    } else {
      shared_result = measure_shared(weights, inputs, shared, rows, cols, repetitions);
      reference_result = measure_reference(weights, inputs, reference, rows, cols, repetitions);
    }
    const bool parity = reference_result.checksum == shared_result.checksum;
    if (!parity) throw std::runtime_error("CPU-E1 checksum mismatch after timing");
    std::cout << std::fixed << std::setprecision(9)
              << "{\"shape\":\"" << shape << "\",\"rows\":" << rows
              << ",\"cols\":" << cols << ",\"pair\":" << pair
              << ",\"order\":\"" << (pair % 2u ? "reference-shared" : "shared-reference")
              << "\",\"repetitions\":" << repetitions
              << ",\"warmup_repetitions\":2"
              << ",\"reference_seconds\":" << reference_result.seconds
              << ",\"shared_seconds\":" << shared_result.seconds
              << ",\"reference_checksum\":" << reference_result.checksum
              << ",\"shared_checksum\":" << shared_result.checksum
              << ",\"parity\":" << (parity ? "true" : "false")
              << ",\"f16c_available\":" << (mm::f16c_available() ? "true" : "false")
              << ",\"cpu\":0,\"threads\":1}\n";
  }
}

std::size_t parse_positive(const char* value, const char* name) {
  try {
    const std::size_t parsed = std::stoull(value);
    if (parsed == 0) throw std::invalid_argument("zero");
    return parsed;
  } catch (...) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t pairs = 6;
    std::size_t repetitions = 64;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--pairs" && index + 1 < argc) pairs = parse_positive(argv[++index], "pairs");
      else if (argument == "--repetitions" && index + 1 < argc) repetitions = parse_positive(argv[++index], "repetitions");
      else throw std::runtime_error("usage: mm_bench_cpu_e1 [--pairs N] [--repetitions N]");
    }
    if (pairs != 6) throw std::runtime_error("CPU-E1 requires exactly six pairs per shape");
    if (!pin_cpu_zero()) throw std::runtime_error("failed to pin benchmark participant to CPU 0");
    run_shape("2432x768", 2432, 768, pairs, repetitions);
    run_shape("768x2432", 768, 2432, pairs, repetitions);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
