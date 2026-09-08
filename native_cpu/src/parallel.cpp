#include "mm_parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#elif (defined(__i386__) || defined(__x86_64__))
#include <immintrin.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace mm {
namespace {

constexpr std::int32_t kUnboundCpu = -1;
constexpr std::int32_t kInvalidWorker = -2;
constexpr auto kWorkerSpinBudget = std::chrono::microseconds(200);
constexpr auto kControllerSpinBudget = std::chrono::milliseconds(2);

inline void relax_cpu() noexcept {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    _mm_pause();
#elif defined(__i386__) || defined(__x86_64__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

struct AffinityGuard {
#if defined(_WIN32)
    GROUP_AFFINITY previous{};
    bool active = false;

    void apply(std::uint32_t cpu) {
        GROUP_AFFINITY requested{};
        requested.Group = 0;
        requested.Mask = static_cast<KAFFINITY>(1) << cpu;
        if (!GetThreadGroupAffinity(GetCurrentThread(), &previous)) {
            throw std::runtime_error("GetThreadGroupAffinity failed: " +
                                     std::to_string(GetLastError()));
        }
        if (!SetThreadGroupAffinity(GetCurrentThread(), &requested, nullptr)) {
            throw std::runtime_error("SetThreadGroupAffinity failed: " +
                                     std::to_string(GetLastError()));
        }
        active = true;
    }

    void restore() noexcept {
        if (active) {
            (void)SetThreadGroupAffinity(GetCurrentThread(), &previous, nullptr);
            active = false;
        }
    }

    ~AffinityGuard() noexcept {
        restore();
    }
#else
    void apply(std::uint32_t) {
        throw std::runtime_error("explicit CPU affinity is unsupported on this platform");
    }

    void restore() noexcept {}
#endif
};

[[nodiscard]] std::size_t weighted_boundary(std::uint64_t cumulative,
                                             std::uint64_t total,
                                             std::size_t rows) noexcept {
    // Model dimensions are bounded 32-bit values, but use long double so the
    // cumulative-weight*rows product cannot wrap before conversion to a row.
    const long double value = (static_cast<long double>(cumulative) *
                               static_cast<long double>(rows)) /
                              static_cast<long double>(total);
    if (value <= 0.0L) return 0;
    if (value >= static_cast<long double>(rows)) return rows;
    return static_cast<std::size_t>(value);
}

} // namespace

struct ParallelTeam::Job {
    enum class Kind : std::uint8_t { F32, F32Row4, F16, Q4 };

    Kind kind = Kind::F32;
    const float* weights = nullptr;
    const std::uint16_t* f16_weights = nullptr;
    const std::uint8_t* packed = nullptr;
    const float* scales = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    KernelMode mode = KernelMode::Scalar;
};

struct ParallelTeam::State {
    struct alignas(64) ProfileSlot { std::uint64_t ns = 0; std::uint64_t calls = 0; };
    struct Range {
        std::size_t begin = 0;
        std::size_t end = 0;
    };

    const std::uint32_t threads;
    const std::array<std::int32_t, kMaxThreads> cpus;
    const std::array<std::uint32_t, kMaxThreads> weights;
    std::vector<std::thread> workers;
    std::array<Range, kMaxThreads> ranges{};
    std::mutex mutex;
    std::condition_variable wake;
    // Job/ranges are written by the synchronous caller, then published with
    // release. Workers acquire the generation before reading either field.
    alignas(64) std::atomic<std::uint64_t> generation{0};
    struct alignas(64) Completion {
        std::atomic<std::uint64_t> epoch{0};
    };
    std::array<Completion, kMaxThreads> completions{};
    std::atomic<bool> active_evaluation{false};
    std::uint32_t ready = 0;
    bool startup_release = false;
    bool startup_failed = false;
    std::atomic<bool> stop{false};
    std::atomic<bool> profile_enabled{false};
    std::array<ProfileSlot, kMaxThreads> participant_slots{};
    std::uint64_t controller_wait_ns = 0;
    std::uint32_t startup_failed_worker = kMaxThreads;
    ParallelTeam::Job job{};
    AffinityGuard caller_affinity;

    State(std::uint32_t count, const std::array<std::int32_t, kMaxThreads>& cpu_values,
          const std::array<std::uint32_t, kMaxThreads>& weight_values)
        : threads(count), cpus(cpu_values), weights(weight_values) {
        workers.reserve(threads > 1u ? threads - 1u : 0u);
    }

    [[nodiscard]] std::uint32_t background_count() const noexcept {
        return threads > 1u ? threads - 1u : 0u;
    }

    void enter_call() {
        if (cpus[0] >= 0) caller_affinity.apply(static_cast<std::uint32_t>(cpus[0]));
        active_evaluation.store(true, std::memory_order_release);
    }

    void leave_call() noexcept {
        // Workers may remain in their short handoff spin while the caller is
        // evaluating. Park them only after the synchronous callback has
        // completed, then restore the caller's previous affinity.
        active_evaluation.store(false, std::memory_order_release);
        caller_affinity.restore();
    }

    static void run_range(const ParallelTeam::Job& job, const Range& range, State* state = nullptr, std::uint32_t participant = 0) noexcept {
        if (range.begin >= range.end) return;
        const bool profiled = state != nullptr && state->profile_enabled.load(std::memory_order_relaxed);
        const auto started = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (job.kind == ParallelTeam::Job::Kind::F32) {
            mm::gemv_f32(job.weights + range.begin * job.cols, job.x,
                         job.y + range.begin, range.end - range.begin,
                         job.cols, job.mode);
            if (profiled) { state->participant_slots[participant].ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()); ++state->participant_slots[participant].calls; }
            return;
        }
        if (job.kind == ParallelTeam::Job::Kind::F32Row4) {
            mm::gemv_f32_row4(job.weights + range.begin * job.cols, job.x,
                              job.y + range.begin, range.end - range.begin,
                              job.cols, job.mode);
            if (profiled) { state->participant_slots[participant].ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()); ++state->participant_slots[participant].calls; }
            return;
        }
        if (job.kind == ParallelTeam::Job::Kind::F16) {
            mm::gemv_f16(job.f16_weights + range.begin * job.cols, job.x,
                         job.y + range.begin, range.end - range.begin,
                         job.cols, job.mode);
            if (profiled) { state->participant_slots[participant].ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()); ++state->participant_slots[participant].calls; }
            return;
        }
        const std::size_t row_bytes = (job.cols + 1u) / 2u;
        const std::size_t groups = (job.cols + 31u) / 32u;
        mm::gemv_q4(job.packed + range.begin * row_bytes,
                    job.scales + range.begin * groups, job.x,
                    job.y + range.begin, range.end - range.begin,
                    job.cols, job.mode);
        if (profiled) { state->participant_slots[participant].ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()); ++state->participant_slots[participant].calls; }
    }

    static void worker_main(State* state, std::uint32_t index) noexcept {
        AffinityGuard affinity;
        bool affinity_ok = true;
        try {
            if (state->cpus[index] >= 0) affinity.apply(static_cast<std::uint32_t>(state->cpus[index]));
        } catch (const std::exception&) {
            affinity_ok = false;
        } catch (...) {
            affinity_ok = false;
        }

        {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (!affinity_ok && !state->startup_failed) {
                state->startup_failed = true;
                state->startup_failed_worker = index;
            }
            ++state->ready;
            state->wake.notify_all();
            state->wake.wait(lock, [state] {
                return state->startup_release || state->stop.load(std::memory_order_acquire);
            });
            if (state->stop.load(std::memory_order_acquire) || state->startup_failed) return;
        }

        // Generation starts at zero and no job is published before startup
        // completes. Do not sample the current value here: the caller may
        // dispatch immediately after start() releases the workers, and a
        // delayed worker must still observe that first generation.
        std::uint64_t seen_generation = 0;
        for (;;) {
            // Most decode GEMVs are short enough that a bounded spin avoids a
            // mutex/syscall round trip. Idle workers then park in C++20's
            // atomic wait and are released by generation.notify_all().
            unsigned spin = 0;
            auto spin_deadline = std::chrono::steady_clock::now() + kWorkerSpinBudget;
            std::uint64_t published = state->generation.load(std::memory_order_acquire);
            while (published == seen_generation && !state->stop.load(std::memory_order_acquire)) {
                if (spin++ < 64u) {
                    relax_cpu();
                } else if (state->active_evaluation.load(std::memory_order_acquire) ||
                           std::chrono::steady_clock::now() < spin_deadline) {
                    spin = 0;
                } else {
                    state->generation.wait(seen_generation, std::memory_order_acquire);
                    spin = 0;
                    spin_deadline = std::chrono::steady_clock::now() + kWorkerSpinBudget;
                }
                published = state->generation.load(std::memory_order_acquire);
            }
            if (state->stop.load(std::memory_order_acquire)) return;

            // The acquire load above synchronizes the descriptor and range
            // writes. No mutex is needed on this hot path.
            const Job job = state->job;
            const Range range = state->ranges[index];
            seen_generation = published;

            run_range(job, range, state, index);

            state->completions[index].epoch.store(seen_generation, std::memory_order_release);
            state->completions[index].epoch.notify_one();
        }
    }

    void start() {
        try {
            for (std::uint32_t index = 1; index < threads; ++index) {
                workers.emplace_back(&State::worker_main, this, index);
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop.store(true, std::memory_order_release);
                startup_release = true;
            }
            wake.notify_all();
            generation.fetch_add(1, std::memory_order_release);
            generation.notify_all();
            for (std::thread& worker : workers) if (worker.joinable()) worker.join();
            throw;
        }

        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [this] { return ready == background_count(); });
        if (startup_failed) {
            const std::string error = startup_failed_worker < threads
                ? "worker " + std::to_string(startup_failed_worker) + " CPU affinity setup failed"
                : "worker startup failed";
            lock.unlock();
            stop_and_join();
            throw std::runtime_error(error);
        }
        startup_release = true;
        lock.unlock();
        wake.notify_all();
    }

    void stop_and_join() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop.store(true, std::memory_order_release);
            startup_release = true;
        }
        wake.notify_all();
        generation.fetch_add(1, std::memory_order_release);
        generation.notify_all();
        for (std::thread& worker : workers) if (worker.joinable()) worker.join();
    }

    ~State() noexcept {
        stop_and_join();
        leave_call();
    }

    void partition(std::size_t rows) noexcept {
        std::uint64_t total = 0;
        for (std::uint32_t index = 0; index < threads; ++index) total += weights[index];
        std::uint64_t cumulative = 0;
        std::size_t begin = 0;
        for (std::uint32_t index = 0; index < threads; ++index) {
            cumulative += weights[index];
            const std::size_t end = index + 1u == threads
                ? rows : std::max(begin, weighted_boundary(cumulative, total, rows));
            ranges[index] = Range{begin, std::min(end, rows)};
            begin = ranges[index].end;
        }
    }

    void dispatch(const ParallelTeam::Job& next_job) {
        partition(next_job.rows);
        job = next_job;
        const std::uint64_t epoch = generation.fetch_add(1, std::memory_order_release) + 1u;
        generation.notify_all();

        // Participant zero is the caller. It computes its disjoint first
        // range while background workers handle ranges 1..N-1.
        const Job caller_job = job;
        run_range(caller_job, ranges[0], this, 0);
        completions[0].epoch.store(epoch, std::memory_order_release);

        const auto wait_started = profile_enabled.load(std::memory_order_relaxed) ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        for (std::uint32_t index = 1; index < threads; ++index) {
            unsigned spin = 0;
            auto spin_deadline = std::chrono::steady_clock::now() + kControllerSpinBudget;
            std::uint64_t completed = completions[index].epoch.load(std::memory_order_acquire);
            while (completed < epoch) {
                if (spin++ < 64u) {
                    relax_cpu();
                } else if (std::chrono::steady_clock::now() < spin_deadline) {
                    spin = 0;
                } else {
                    completions[index].epoch.wait(completed, std::memory_order_acquire);
                    spin = 0;
                    spin_deadline = std::chrono::steady_clock::now() + kControllerSpinBudget;
                }
                completed = completions[index].epoch.load(std::memory_order_acquire);
            }
        }
        if (profile_enabled.load(std::memory_order_relaxed)) controller_wait_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_started).count());
    }
};

ParallelTeam::ParallelTeam() noexcept {
    cpus_.fill(kUnboundCpu);
    weights_.fill(1u);
}

ParallelTeam::~ParallelTeam() noexcept = default;

void ParallelTeam::configure(std::uint32_t threads, const std::uint32_t* cpu_indices,
                             const std::uint32_t* row_weights) {
    if (threads == 0 || threads > kMaxThreads) {
        throw std::runtime_error("thread count must be in [1, 64]");
    }
    std::array<std::int32_t, kMaxThreads> next_cpus{};
    std::array<std::uint32_t, kMaxThreads> next_weights{};
    next_cpus.fill(kUnboundCpu);
    next_weights.fill(1u);

    if (row_weights != nullptr) {
        for (std::uint32_t index = 0; index < threads; ++index) {
            if (row_weights[index] == 0) {
                throw std::runtime_error("row weights must be positive");
            }
            next_weights[index] = row_weights[index];
        }
    }
    if (cpu_indices != nullptr) {
#if defined(_WIN32)
        const DWORD active = GetActiveProcessorCount(0);
        if (active == 0 || active > sizeof(KAFFINITY) * 8u) {
            throw std::runtime_error("Windows processor group 0 is unavailable");
        }
        for (std::uint32_t index = 0; index < threads; ++index) {
            if (cpu_indices[index] >= active) {
                throw std::runtime_error("CPU index is not active in Windows group 0: " +
                                         std::to_string(cpu_indices[index]));
            }
            for (std::uint32_t prior = 0; prior < index; ++prior) {
                if (cpu_indices[prior] == cpu_indices[index]) {
                    throw std::runtime_error("duplicate CPU index");
                }
            }
            next_cpus[index] = static_cast<std::int32_t>(cpu_indices[index]);
        }
        // Probe participant zero on the caller before constructing/swapping
        // a worker team. The guard restores the caller before configure()
        // returns, so a failed probe leaves the old configuration untouched.
        AffinityGuard caller_probe;
        caller_probe.apply(static_cast<std::uint32_t>(next_cpus[0]));
#else
        (void)cpu_indices;
        throw std::runtime_error("explicit CPU affinity is unsupported on this platform");
#endif
    }

    std::unique_ptr<State> candidate;
    if (threads > 1 || cpu_indices != nullptr) {
        candidate = std::make_unique<State>(threads, next_cpus, next_weights);
        candidate->profile_enabled.store(profile_enabled_, std::memory_order_release);
        candidate->start();
    }

    state_.swap(candidate);
    threads_ = threads;
    cpus_ = next_cpus;
    weights_ = next_weights;
}

void ParallelTeam::execute(void (*work)(void*), void* context) {
    if (work == nullptr) throw std::runtime_error("parallel operation callback is null");
    if (state_ != nullptr) state_->enter_call();
    try {
        work(context);
    } catch (...) {
        if (state_ != nullptr) state_->leave_call();
        throw;
    }
    if (state_ != nullptr) state_->leave_call();
}

std::int32_t ParallelTeam::thread_cpu(std::uint32_t worker) const noexcept {
    return worker < threads_ ? cpus_[worker] : kInvalidWorker;
}

std::uint32_t ParallelTeam::thread_weight(std::uint32_t worker) const noexcept {
    return worker < threads_ ? weights_[worker] : 0u;
}

void ParallelTeam::dispatch(const Job& job) {
    if (state_ == nullptr) {
        const auto started = profile_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (job.kind == Job::Kind::F32) {
            mm::gemv_f32(job.weights, job.x, job.y, job.rows, job.cols, job.mode);
        } else if (job.kind == Job::Kind::F32Row4) {
            mm::gemv_f32_row4(job.weights, job.x, job.y, job.rows, job.cols, job.mode);
        } else if (job.kind == Job::Kind::F16) {
            mm::gemv_f16(job.f16_weights, job.x, job.y, job.rows, job.cols, job.mode);
        } else {
            mm::gemv_q4(job.packed, job.scales, job.x, job.y, job.rows, job.cols, job.mode);
        }
        if (profile_enabled_) { diagnostics_.participant_compute_ns[0] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()); diagnostics_.participant_compute_calls[0]++; }
        return;
    }
    state_->dispatch(job);
}

void ParallelTeam::configure_profile(bool enabled) noexcept {
    profile_enabled_ = enabled;
    if (state_) state_->profile_enabled.store(enabled, std::memory_order_release);
}

void ParallelTeam::reset_profile_stats() noexcept {
    diagnostics_ = {};
    if (state_) {
        for (auto& slot : state_->participant_slots) slot = {};
        state_->controller_wait_ns = 0;
    }
}

ParallelTeam::Diagnostics ParallelTeam::profile_stats() const noexcept {
    Diagnostics result = diagnostics_;
    if (state_) {
        for (std::size_t i = 0; i < kMaxThreads; ++i) { result.participant_compute_ns[i] = state_->participant_slots[i].ns; result.participant_compute_calls[i] = state_->participant_slots[i].calls; }
        result.controller_wait_ns = state_->controller_wait_ns;
    }
    return result;
}

void ParallelTeam::gemv_f32(const float* weights, const float* x, float* y,
                            std::size_t rows, std::size_t cols, KernelMode mode) {
    Job job;
    job.kind = Job::Kind::F32;
    job.weights = weights;
    job.x = x;
    job.y = y;
    job.rows = rows;
    job.cols = cols;
    job.mode = mode;
    dispatch(job);
}

void ParallelTeam::gemv_f32_row4(const float* weights, const float* x, float* y,
                                 std::size_t rows, std::size_t cols,
                                 KernelMode mode) {
    Job job;
    job.kind = Job::Kind::F32Row4;
    job.weights = weights;
    job.x = x;
    job.y = y;
    job.rows = rows;
    job.cols = cols;
    job.mode = mode;
    dispatch(job);
}

void ParallelTeam::gemv_f16(const std::uint16_t* weights, const float* x,
                             float* y, std::size_t rows, std::size_t cols,
                             KernelMode mode) {
    Job job;
    job.kind = Job::Kind::F16;
    job.f16_weights = weights;
    job.x = x;
    job.y = y;
    job.rows = rows;
    job.cols = cols;
    job.mode = mode;
    dispatch(job);
}

void ParallelTeam::gemv_q4(const std::uint8_t* packed, const float* scales,
                           const float* x, float* y, std::size_t rows,
                           std::size_t cols, KernelMode mode) {
    Job job;
    job.kind = Job::Kind::Q4;
    job.packed = packed;
    job.scales = scales;
    job.x = x;
    job.y = y;
    job.rows = rows;
    job.cols = cols;
    job.mode = mode;
    dispatch(job);
}

} // namespace mm
