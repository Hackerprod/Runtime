#pragma once

#include "mm_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mm {

// A persistent row-sharding team for GEMV. Configuration is synchronous and
// transactional; callers must not configure or execute one team concurrently.
class ParallelTeam final {
public:
    static constexpr std::uint32_t kMaxThreads = 64;
    struct Diagnostics {
        std::array<std::uint64_t, kMaxThreads> participant_compute_ns{};
        std::array<std::uint64_t, kMaxThreads> participant_compute_calls{};
        std::uint64_t controller_wait_ns = 0;
    };
    ParallelTeam() noexcept;
    ~ParallelTeam() noexcept;

    ParallelTeam(const ParallelTeam&) = delete;
    ParallelTeam& operator=(const ParallelTeam&) = delete;

    void configure(std::uint32_t threads, const std::uint32_t* cpu_indices,
                   const std::uint32_t* row_weights);

    [[nodiscard]] std::uint32_t thread_count() const noexcept { return threads_; }
    [[nodiscard]] std::int32_t thread_cpu(std::uint32_t worker) const noexcept;
    [[nodiscard]] std::uint32_t thread_weight(std::uint32_t worker) const noexcept;

    // Run one synchronous model operation while temporarily applying the
    // configured participant-0 affinity to the caller. The affinity is
    // restored on normal and exceptional return. The callback must not
    // configure or execute this team recursively.
    void execute(void (*work)(void*), void* context);

    void gemv_f32(const float* weights, const float* x, float* y,
                  std::size_t rows, std::size_t cols, KernelMode mode);
    void gemv_q4(const std::uint8_t* packed, const float* scales, const float* x,
                 float* y, std::size_t rows, std::size_t cols, KernelMode mode);
    void configure_profile(bool enabled) noexcept;
    void reset_profile_stats() noexcept;
    [[nodiscard]] Diagnostics profile_stats() const noexcept;

private:
    struct Job;
    struct State;

    void dispatch(const Job& job);

    std::unique_ptr<State> state_;
    std::uint32_t threads_ = 1;
    std::array<std::int32_t, kMaxThreads> cpus_{};
    std::array<std::uint32_t, kMaxThreads> weights_{};
    bool profile_enabled_ = false;
    Diagnostics diagnostics_{};
};

} // namespace mm
