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
#define MM_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define MM_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define MM_NOINLINE __attribute__((noinline))
#else
#define MM_TARGET_AVX2
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
