#include "mm_kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

void check_close(const std::vector<float>& a, const std::vector<float>& b,
                 float tolerance) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    assert(std::isfinite(a[i]));
    assert(std::abs(a[i] - b[i]) <= tolerance * (1.0f + std::abs(b[i])));
  }
}

void test_f32(std::size_t rows, std::size_t cols) {
  std::vector<float> weights(rows * cols), x(cols), scalar(rows), automatic(rows);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<float>((static_cast<int>(i * 13u % 29u) - 14)) * 0.03125f;
  for (std::size_t i = 0; i < cols; ++i)
    x[i] = std::sin(static_cast<float>(i) * 0.17f);
  mm::gemv_f32(weights.data(), x.data(), scalar.data(), rows, cols,
               mm::KernelMode::Scalar);
  mm::gemv_f32(weights.data(), x.data(), automatic.data(), rows, cols,
               mm::KernelMode::Auto);
  check_close(scalar, automatic, 3e-5f);
}

void test_f32_row4_exact(std::size_t rows, std::size_t cols) {
  std::vector<float> weights(rows * cols), baseline(rows), row4(rows), x(cols);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<float>((static_cast<int>(i * 19u % 31u) - 15)) * 0.03125f;
  for (std::size_t i = 0; i < cols; ++i)
    x[i] = std::sin(static_cast<float>(i) * 0.13f) + 0.01f;
  for (mm::KernelMode mode : {mm::KernelMode::Scalar, mm::KernelMode::Auto}) {
    std::fill(baseline.begin(), baseline.end(), 0.0f);
    std::fill(row4.begin(), row4.end(), 0.0f);
    mm::gemv_f32(weights.data(), x.data(), baseline.data(), rows, cols, mode);
    mm::gemv_f32_row4(weights.data(), x.data(), row4.data(), rows, cols, mode);
    assert(std::memcmp(baseline.data(), row4.data(), rows * sizeof(float)) == 0);
  }
}

void test_q4(std::size_t rows, std::size_t cols) {
  const std::size_t row_bytes = (cols + 1u) / 2u;
  const std::size_t groups = (cols + 31u) / 32u;
  std::vector<std::uint8_t> packed(rows * row_bytes, 0);
  std::vector<float> scales(rows * groups), x(cols), scalar(rows), automatic(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t group = 0; group < groups; ++group)
      scales[row * groups + group] = 0.125f + 0.03125f * static_cast<float>((row + group) % 5u);
    for (std::size_t col = 0; col < cols; ++col) {
      const std::uint8_t nibble = static_cast<std::uint8_t>((row * 3u + col * 5u) % 16u);
      auto& byte = packed[row * row_bytes + (col >> 1u)];
      if ((col & 1u) == 0u) byte = static_cast<std::uint8_t>((byte & 0xf0u) | nibble);
      else byte = static_cast<std::uint8_t>((byte & 0x0fu) | (nibble << 4u));
    }
  }
  for (std::size_t col = 0; col < cols; ++col) x[col] = std::cos(static_cast<float>(col) * 0.11f);
  mm::gemv_q4(packed.data(), scales.data(), x.data(), scalar.data(), rows, cols,
              mm::KernelMode::Scalar);
  mm::gemv_q4(packed.data(), scales.data(), x.data(), automatic.data(), rows, cols,
              mm::KernelMode::Auto);
  check_close(scalar, automatic, 4e-5f);
}

void test_q4_zero_rows() {
  constexpr std::size_t rows = 2;
  constexpr std::size_t cols = 65;
  const std::size_t row_bytes = (cols + 1u) / 2u;
  const std::size_t groups = (cols + 31u) / 32u;
  std::vector<std::uint8_t> packed(rows * row_bytes, 0x88u);
  std::vector<float> scales(rows * groups, 3.0f);
  std::vector<float> x(cols, 7.0f), output(rows, 123.0f);
  mm::gemv_q4(packed.data(), scales.data(), x.data(), output.data(), rows, cols,
              mm::KernelMode::Auto);
  for (float value : output) assert(value == 0.0f);
}

void test_q4_known_nibbles() {
  // Values are [-8, -7, 0, 7, 0] with the final high nibble unused.
  const std::uint8_t packed[] = {0x10u, 0xf8u, 0x88u};
  const float scales[] = {1.0f};
  const float x[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float output = 0.0f;
  mm::gemv_q4(packed, scales, x, &output, 1, 5, mm::KernelMode::Scalar);
  assert(output == -8.0f);
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void test_f16_storage_exact() {
  const float exact[] = {
      0.0f, -0.0f, 1.0f, -2.0f, 0.5f,
      std::ldexp(1.0f, -24), 65504.0f, -65504.0f};
  for (float value : exact) {
    const auto half = mm::f32_to_f16(value);
    const float restored = mm::f16_to_f32(half);
    assert(float_bits(restored) == float_bits(value));
  }
  // The activation contract is exact, not an error-tolerance check.
  assert(float_bits(mm::f16_to_f32(mm::f32_to_f16(0.1f))) != float_bits(0.1f));

  constexpr std::size_t rows = 5, cols = 9;
  std::vector<float> weights(rows * cols), input(cols), reference(rows), compact(rows);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    static constexpr float values[] = {0.0f, -0.0f, 0.5f, -1.0f, 2.0f,
                                       0.25f, -0.125f, 4.0f, -8.0f};
    weights[i] = values[i % (sizeof(values) / sizeof(values[0]))];
  }
  for (std::size_t i = 0; i < cols; ++i) input[i] = std::sin(static_cast<float>(i) * 0.37f);
  std::vector<std::uint16_t> packed(weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i) packed[i] = mm::f32_to_f16(weights[i]);
  for (mm::KernelMode mode : {mm::KernelMode::Scalar, mm::KernelMode::Auto}) {
    mm::gemv_f32(weights.data(), input.data(), reference.data(), rows, cols, mode);
    mm::gemv_f16(packed.data(), input.data(), compact.data(), rows, cols, mode);
    if (mode == mm::KernelMode::Scalar || mm::f16c_available()) {
      assert(std::memcmp(reference.data(), compact.data(), rows * sizeof(float)) == 0);
    }
  }
}

}  // namespace

int main() {
  test_f32(768, 384);
  test_f32(2, 2432);
  for (std::size_t rows : {1u, 3u, 4u, 5u, 7u, 8u, 9u}) {
    for (std::size_t cols : {1u, 7u, 8u, 9u, 2432u}) test_f32_row4_exact(rows, cols);
  }
  test_q4(3, 33);
  test_q4(3, 65);
  test_q4(4, 2432);
  test_q4_zero_rows();
  test_q4_known_nibbles();
  test_f16_storage_exact();
  std::cout << "mm kernels tests passed (scalar reference and auto dispatch)\n";
  return 0;
}
