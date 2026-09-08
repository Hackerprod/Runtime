#include "mm_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace mm {
namespace {

#if defined(_MSC_VER)
#define MM_TARGET_AVX2
#define MM_TARGET_F16C
#define MM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define MM_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define MM_TARGET_F16C __attribute__((target("avx2,fma,f16c")))
#define MM_NOINLINE __attribute__((noinline))
#else
#define MM_TARGET_AVX2
#define MM_TARGET_F16C
#define MM_NOINLINE
#endif

bool detect_avx2_fma() noexcept {
#if defined(_MSC_VER)
  int regs[4]{};
  __cpuidex(regs, 0, 0);
  const int max_leaf = regs[0];
  if (max_leaf < 1) return false;
  __cpuidex(regs, 1, 0);
  constexpr int osxsave = 1 << 27;
  constexpr int avx = 1 << 28;
  constexpr int fma = 1 << 12;
  if ((regs[2] & (osxsave | avx | fma)) != (osxsave | avx | fma)) {
    return false;
  }
  if ((_xgetbv(0) & 0x6u) != 0x6u || max_leaf < 7) return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
  // __builtin_cpu_supports also accounts for OS AVX state on supported
  // x86 targets and avoids executing an intrinsic before the runtime check.
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

bool detect_f16c() noexcept {
#if defined(_MSC_VER)
  int regs[4]{};
  __cpuidex(regs, 0, 0);
  const int max_leaf = regs[0];
  if (max_leaf < 1) return false;
  __cpuidex(regs, 1, 0);
  constexpr int osxsave = 1 << 27;
  constexpr int avx = 1 << 28;
  constexpr int f16c = 1 << 29;
  if ((regs[2] & (osxsave | avx | f16c)) != (osxsave | avx | f16c)) return false;
  return (_xgetbv(0) & 0x6u) == 0x6u;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx");
#else
  return false;
#endif
}

std::uint32_t float_bits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(std::uint32_t bits) noexcept {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint16_t f32_to_f16_impl(float value) noexcept {
  const std::uint32_t bits = float_bits(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    if (mantissa == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
    mantissa >>= 13;
    return static_cast<std::uint16_t>(sign | 0x7c00u | mantissa | (mantissa == 0));
  }
  const int unbiased = static_cast<int>(exponent) - 127;
  int half_exponent = unbiased + 15;
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000u;
    const int shift = 14 - half_exponent;
    std::uint32_t result = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (result & 1u))) ++result;
    return static_cast<std::uint16_t>(sign | result);
  }
  if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  std::uint32_t result = mantissa >> 13;
  const std::uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u))) {
    ++result;
    if (result == 0x400u) {
      result = 0;
      ++half_exponent;
      if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10) | result);
}

float f16_to_f32_impl(std::uint16_t value) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  int exponent = static_cast<int>((value >> 10) & 0x1fu);
  std::uint32_t mantissa = value & 0x3ffu;
  if (exponent == 0) {
    if (mantissa == 0) return bits_float(sign);
    exponent = 1;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    mantissa &= 0x3ffu;
    return bits_float(sign | (static_cast<std::uint32_t>(exponent + 127 - 15) << 23) | (mantissa << 13));
  }
  if (exponent == 31) return bits_float(sign | 0x7f800000u | (mantissa << 13));
  return bits_float(sign | (static_cast<std::uint32_t>(exponent + 127 - 15) << 23) | (mantissa << 13));
}

inline bool use_avx2(KernelMode mode) noexcept {
  return mode == KernelMode::Auto && avx2_fma_available();
}

MM_NOINLINE void gemv_f32_scalar(const float* weights, const float* x,
                                 float* y, std::size_t rows,
                                 std::size_t cols) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* w = weights + row * cols;
    float sum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) sum += w[col] * x[col];
    y[row] = sum;
  }
}

MM_NOINLINE void gemv_q4_scalar(const std::uint8_t* packed,
                                const float* scales, const float* x,
                                float* y, std::size_t rows,
                                std::size_t cols) noexcept {
  const std::size_t row_bytes = (cols + 1u) / 2u;
  const std::size_t groups = (cols + 31u) / 32u;
  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint8_t* row_packed = packed + row * row_bytes;
    const float* row_scales = scales + row * groups;
    float sum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
      const std::uint8_t byte = row_packed[col >> 1u];
      const std::uint8_t nibble = (col & 1u) == 0u ? byte & 0x0fu : byte >> 4u;
      sum += x[col] * (static_cast<float>(static_cast<int>(nibble) - 8) *
                       row_scales[col >> 5u]);
    }
    y[row] = sum;
  }
}

MM_TARGET_AVX2 inline __m256 q4_to_float8(__m128i values, float scale) noexcept {
  const __m128i q16 = _mm_cvtepi8_epi16(values);
  const __m128i q32_lo = _mm_cvtepi16_epi32(q16);
  const __m128i q32_hi =
      _mm_cvtepi16_epi32(_mm_srli_si128(q16, sizeof(std::int32_t) * 2));
  const __m128 scale4 = _mm_set1_ps(scale);
  return _mm256_set_m128(_mm_mul_ps(_mm_cvtepi32_ps(q32_hi), scale4),
                         _mm_mul_ps(_mm_cvtepi32_ps(q32_lo), scale4));
}

MM_TARGET_AVX2 inline float horizontal_sum(__m256 value) noexcept {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                          _mm256_extractf128_ps(value, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

MM_TARGET_AVX2 MM_NOINLINE void gemv_f32_avx2(
    const float* weights, const float* x, float* y, std::size_t rows,
    std::size_t cols) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* w = weights + row * cols;
    __m256 acc = _mm256_setzero_ps();
    std::size_t col = 0;
    for (; col + 8u <= cols; col += 8u) {
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + col),
                            _mm256_loadu_ps(x + col), acc);
    }
    float sum = horizontal_sum(acc);
    for (; col < cols; ++col) sum += w[col] * x[col];
    y[row] = sum;
  }
}

MM_TARGET_AVX2 MM_NOINLINE void gemv_f32_row4_avx2(
    const float* weights, const float* x, float* y, std::size_t rows,
    std::size_t cols) noexcept {
  std::size_t row = 0;
  for (; row + 4u <= rows; row += 4u) {
    const float* w0 = weights + (row + 0u) * cols;
    const float* w1 = weights + (row + 1u) * cols;
    const float* w2 = weights + (row + 2u) * cols;
    const float* w3 = weights + (row + 3u) * cols;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t col = 0;
    for (; col + 8u <= cols; col += 8u) {
      // Load each activation fragment once and reuse it across four rows.
      const __m256 input = _mm256_loadu_ps(x + col);
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + col), input, acc0);
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + col), input, acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + col), input, acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + col), input, acc3);
    }
    float sum0 = horizontal_sum(acc0);
    float sum1 = horizontal_sum(acc1);
    float sum2 = horizontal_sum(acc2);
    float sum3 = horizontal_sum(acc3);
    for (; col < cols; ++col) {
      sum0 += w0[col] * x[col];
      sum1 += w1[col] * x[col];
      sum2 += w2[col] * x[col];
      sum3 += w3[col] * x[col];
    }
    y[row + 0u] = sum0;
    y[row + 1u] = sum1;
    y[row + 2u] = sum2;
    y[row + 3u] = sum3;
  }
  if (row < rows) {
    gemv_f32_avx2(weights + row * cols, x, y + row, rows - row, cols);
  }
}

MM_TARGET_F16C MM_NOINLINE void gemv_f16_avx2(
    const std::uint16_t* weights, const float* x, float* y,
    std::size_t rows, std::size_t cols) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint16_t* w = weights + row * cols;
    __m256 acc = _mm256_setzero_ps();
    std::size_t col = 0;
    for (; col + 8u <= cols; col += 8u) {
      const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + col));
      const __m256 converted = _mm256_cvtph_ps(packed);
      acc = _mm256_fmadd_ps(converted, _mm256_loadu_ps(x + col), acc);
    }
    float sum = horizontal_sum(acc);
    for (; col < cols; ++col) sum += f16_to_f32_impl(w[col]) * x[col];
    y[row] = sum;
  }
}

MM_TARGET_F16C MM_NOINLINE void gemv_f16_x4_avx2(
    const std::uint16_t* weights, const float* x0, float* y0,
    const float* x1, float* y1, const float* x2, float* y2,
    const float* x3, float* y3, std::size_t rows,
    std::size_t cols) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint16_t* w = weights + row * cols;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t col = 0;
    for (; col + 8u <= cols; col += 8u) {
      // Convert each weight fragment once, then reuse it for all four
      // independent inputs. Each accumulator follows gemv_f16's arithmetic.
      const __m128i packed = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(w + col));
      const __m256 converted = _mm256_cvtph_ps(packed);
      acc0 = _mm256_fmadd_ps(converted, _mm256_loadu_ps(x0 + col), acc0);
      acc1 = _mm256_fmadd_ps(converted, _mm256_loadu_ps(x1 + col), acc1);
      acc2 = _mm256_fmadd_ps(converted, _mm256_loadu_ps(x2 + col), acc2);
      acc3 = _mm256_fmadd_ps(converted, _mm256_loadu_ps(x3 + col), acc3);
    }
    float sum0 = horizontal_sum(acc0);
    float sum1 = horizontal_sum(acc1);
    float sum2 = horizontal_sum(acc2);
    float sum3 = horizontal_sum(acc3);
    for (; col < cols; ++col) {
      const float converted = f16_to_f32_impl(w[col]);
      sum0 += converted * x0[col];
      sum1 += converted * x1[col];
      sum2 += converted * x2[col];
      sum3 += converted * x3[col];
    }
    y0[row] = sum0;
    y1[row] = sum1;
    y2[row] = sum2;
    y3[row] = sum3;
  }
}

MM_TARGET_AVX2 MM_NOINLINE void gemv_q4_avx2(
    const std::uint8_t* packed, const float* scales, const float* x,
    float* y, std::size_t rows, std::size_t cols) noexcept {
  const std::size_t row_bytes = (cols + 1u) / 2u;
  const std::size_t groups = (cols + 31u) / 32u;
  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i bias = _mm_set1_epi8(8);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint8_t* row_packed = packed + row * row_bytes;
    const float* row_scales = scales + row * groups;
    float sum = 0.0f;
    std::size_t col = 0;
    for (; col + 32u <= cols; col += 32u) {
      const __m128i bytes = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(row_packed + (col >> 1u)));
      const __m128i low = _mm_and_si128(bytes, mask);
      const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
      const __m128i qlo = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), bias);
      const __m128i qhi = _mm_sub_epi8(_mm_unpackhi_epi8(low, high), bias);
      const float scale = row_scales[col >> 5u];
      __m256 acc = _mm256_setzero_ps();
      acc = _mm256_fmadd_ps(q4_to_float8(qlo, scale),
                            _mm256_loadu_ps(x + col), acc);
      acc = _mm256_fmadd_ps(q4_to_float8(_mm_srli_si128(qlo, 8), scale),
                            _mm256_loadu_ps(x + col + 8), acc);
      acc = _mm256_fmadd_ps(q4_to_float8(qhi, scale),
                            _mm256_loadu_ps(x + col + 16), acc);
      acc = _mm256_fmadd_ps(q4_to_float8(_mm_srli_si128(qhi, 8), scale),
                            _mm256_loadu_ps(x + col + 24), acc);
      sum += horizontal_sum(acc);
    }
    for (; col < cols; ++col) {
      const std::uint8_t byte = row_packed[col >> 1u];
      const std::uint8_t nibble = (col & 1u) == 0u ? byte & 0x0fu : byte >> 4u;
      sum += x[col] * (static_cast<float>(static_cast<int>(nibble) - 8) *
                       row_scales[col >> 5u]);
    }
    y[row] = sum;
  }
}

}  // namespace

bool avx2_fma_available() noexcept {
  static const bool available = detect_avx2_fma();
  return available;
}

bool f16c_available() noexcept {
  static const bool available = detect_f16c() && avx2_fma_available();
  return available;
}

std::uint16_t f32_to_f16(float value) noexcept { return f32_to_f16_impl(value); }

float f16_to_f32(std::uint16_t value) noexcept { return f16_to_f32_impl(value); }

const char* kernel_name(KernelMode mode) noexcept {
  if (mode == KernelMode::Auto) return avx2_fma_available() ? "avx2" : "scalar";
  return "scalar";
}

void gemv_f32(const float* weights, const float* x, float* y, std::size_t rows,
              std::size_t cols, KernelMode mode) noexcept {
  if (weights == nullptr || x == nullptr || y == nullptr || rows == 0 ||
      cols == 0) {
    return;
  }
  if (use_avx2(mode)) {
    gemv_f32_avx2(weights, x, y, rows, cols);
  } else {
    gemv_f32_scalar(weights, x, y, rows, cols);
  }
}

void gemv_f32_row4(const float* weights, const float* x, float* y,
                   std::size_t rows, std::size_t cols,
                   KernelMode mode) noexcept {
  if (weights == nullptr || x == nullptr || y == nullptr || rows == 0 ||
      cols == 0) {
    return;
  }
  if (use_avx2(mode)) {
    gemv_f32_row4_avx2(weights, x, y, rows, cols);
  } else {
    gemv_f32_scalar(weights, x, y, rows, cols);
  }
}

void gemv_f16(const std::uint16_t* weights, const float* x, float* y,
              std::size_t rows, std::size_t cols, KernelMode mode) noexcept {
  if (weights == nullptr || x == nullptr || y == nullptr || rows == 0 || cols == 0) return;
  if (mode == KernelMode::Auto && f16c_available()) {
    gemv_f16_avx2(weights, x, y, rows, cols);
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) sum += f16_to_f32_impl(weights[row * cols + col]) * x[col];
    y[row] = sum;
  }
}

void gemv_f16_x4(const std::uint16_t* weights,
                 const float* x0, float* y0,
                 const float* x1, float* y1,
                 const float* x2, float* y2,
                 const float* x3, float* y3,
                 std::size_t rows, std::size_t cols,
                 KernelMode mode) noexcept {
  if (weights == nullptr || x0 == nullptr || y0 == nullptr ||
      x1 == nullptr || y1 == nullptr || x2 == nullptr || y2 == nullptr ||
      x3 == nullptr || y3 == nullptr || rows == 0 || cols == 0) {
    return;
  }
  if (mode == KernelMode::Auto && f16c_available()) {
    gemv_f16_x4_avx2(weights, x0, y0, x1, y1, x2, y2, x3, y3, rows, cols);
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
      const float converted = f16_to_f32_impl(weights[row * cols + col]);
      sum0 += converted * x0[col];
      sum1 += converted * x1[col];
      sum2 += converted * x2[col];
      sum3 += converted * x3[col];
    }
    y0[row] = sum0;
    y1[row] = sum1;
    y2[row] = sum2;
    y3[row] = sum3;
  }
}

void gemv_q4(const std::uint8_t* packed, const float* scales, const float* x,
             float* y, std::size_t rows, std::size_t cols,
             KernelMode mode) noexcept {
  if (packed == nullptr || scales == nullptr || x == nullptr || y == nullptr ||
      rows == 0 || cols == 0) {
    return;
  }
  if (use_avx2(mode)) {
    gemv_q4_avx2(packed, scales, x, y, rows, cols);
  } else {
    gemv_q4_scalar(packed, scales, x, y, rows, cols);
  }
}

}  // namespace mm
