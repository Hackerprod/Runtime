#pragma once

#include <cstddef>
#include <cstdint>

namespace mm {

enum class KernelMode : int {
  Scalar = 0,
  Auto = 1,
};

// Reports whether AVX2 and FMA are available and the OS has enabled the
// register state required to execute them.
bool avx2_fma_available() noexcept;

// Reports whether the OS and CPU expose the AVX2/FMA/F16C state required by
// the compact-FP16 storage kernel.
bool f16c_available() noexcept;

// IEEE-754 conversions used to validate the compact weight representation.
// The conversion is round-to-nearest-even and does not require F16C.
std::uint16_t f32_to_f16(float value) noexcept;
float f16_to_f32(std::uint16_t value) noexcept;

// Returns the implementation selected by mode ("scalar" or "avx2").
const char* kernel_name(KernelMode mode) noexcept;

// Row-major FP32 matrix-vector product: y = weights * x.
// The caller owns and validates all buffers and dimensions. Null or zero-size
// inputs are treated as no-ops so the ABI never throws across a boundary.
void gemv_f32(const float* weights, const float* x, float* y,
              std::size_t rows, std::size_t cols,
              KernelMode mode = KernelMode::Auto) noexcept;

// Row-major FP32 matrix-vector product using four independent row accumulators
// per block. The arithmetic order within each row matches gemv_f32; rows that
// do not complete a block use the existing implementation.
void gemv_f32_row4(const float* weights, const float* x, float* y,
                   std::size_t rows, std::size_t cols,
                   KernelMode mode = KernelMode::Auto) noexcept;

// Row-major FP16-storage matrix-vector product. Weights are converted to
// FP32 in vector registers and accumulated with the same FP32 FMA/reduction
// order as gemv_f32. Callers must dispatch this only when f16c_available().
void gemv_f16(const std::uint16_t* weights, const float* x, float* y,
              std::size_t rows, std::size_t cols,
              KernelMode mode = KernelMode::Auto) noexcept;

// Row-major symmetric packed-Q4 matrix-vector product. Each row has
// ceil(cols / 2) bytes and ceil(cols / 32) FP32 scales. The even element is
// stored in the low nibble; decoded value is (nibble - 8) * scale.
void gemv_q4(const std::uint8_t* packed, const float* scales, const float* x,
             float* y, std::size_t rows, std::size_t cols,
             KernelMode mode = KernelMode::Auto) noexcept;

}  // namespace mm
