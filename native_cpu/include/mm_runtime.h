#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(MM_RUNTIME_BUILD)
#    define MM_RUNTIME_API __declspec(dllexport)
#  else
#    define MM_RUNTIME_API __declspec(dllimport)
#  endif
#else
#  define MM_RUNTIME_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Load and validate an MMCPU001 dense MiniMind tensor file.
 * max_context must be in [1, max_position_embeddings]. kernel_mode is
 * 0 (scalar) or 1 (auto). On failure, error receives a NUL-terminated
 * diagnostic when a buffer is supplied. */
MM_RUNTIME_API void* mm_load(const char* path, uint32_t max_context,
                             int kernel_mode, char* error, size_t error_cap);
MM_RUNTIME_API void mm_free(void* runtime);
MM_RUNTIME_API int mm_reset(void* runtime);
MM_RUNTIME_API uint32_t mm_vocab_size(void* runtime);
MM_RUNTIME_API uint32_t mm_position(void* runtime);
MM_RUNTIME_API const char* mm_backend(void* runtime);

/* Configure the persistent row-sharding GEMV worker team. threads is in
 * [1, 64] total participants. Participant 0 is the caller during mm_eval;
 * remaining participants are persistent workers. A null CPU array leaves
 * participants unbound; a null weight array uses equal positive row weights.
 * On Windows, CPU indices are logical indices in processor group 0 (not
 * global system indices); participant 0 is scoped to each mm_eval and then
 * restored. Returns 0 on success and -1 on validation or worker-startup
 * failure, preserving the previous config. */
MM_RUNTIME_API int mm_configure_threads(void* runtime, uint32_t threads,
                                         const uint32_t* cpu_indices,
                                         const uint32_t* row_weights,
                                         char* error, size_t error_cap);
MM_RUNTIME_API uint32_t mm_thread_count(void* runtime);
MM_RUNTIME_API int32_t mm_thread_cpu(void* runtime, uint32_t worker);
MM_RUNTIME_API uint32_t mm_thread_weight(void* runtime, uint32_t worker);

/* Evaluate count sequential tokens and write only the final token logits.
 * All token IDs and capacity are checked before cache mutation. */
MM_RUNTIME_API int mm_eval(void* runtime, const int32_t* token_ids,
                           size_t count, float* last_logits,
                           size_t logits_capacity, char* error,
                           size_t error_cap);

#ifdef __cplusplus
}
#endif
