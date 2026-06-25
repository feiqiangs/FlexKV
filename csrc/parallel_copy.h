/*
 * [P1] Parallel CPU gather/scatter for CE-mode KV transfer.
 *
 * Long-context + CE-only loads are bottlenecked by the single-threaded
 * std::memcpy gather/scatter loops inside transfer.cu Path 2. These helpers
 * fan the per-block copies across a persistent CPU thread pool and use
 * non-temporal (streaming) stores on the write side to avoid RFO traffic and
 * cache pollution for the write-once staging / KV-pool destinations.
 *
 * Declarations only (no SIMD intrinsics) so this header is safe to include
 * from .cu translation units compiled by nvcc. The implementation lives in
 * parallel_copy.cpp and is compiled by the host C++ compiler.
 *
 * Env knobs:
 *   FLEXKV_PARALLEL_GATHER=0   disable threading (single-thread fallback)
 *   FLEXKV_GATHER_THREADS=N    total participating threads (incl. caller);
 *                              default = max(1, min(hw,16)/2)
 *   FLEXKV_GATHER_NT=0         disable non-temporal stores (use memcpy)
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace flexkv {

// gather: staging[k*chunk_bytes ..] <- src_base + block_ids[k]*block_stride_bytes
//   (scattered read, contiguous write). NT store on staging.
void parallel_gather_blocks(void *staging, const void *src_base,
                            const int64_t *block_ids,
                            int64_t block_stride_bytes, int64_t chunk_bytes,
                            int num_blocks);

// scatter: dst_base + block_ids[k]*block_stride_bytes <- staging[k*chunk_bytes ..]
//   (contiguous read, scattered write). NT store on dst_base.
void parallel_scatter_blocks(void *dst_base, const int64_t *block_ids,
                             int64_t block_stride_bytes, const void *staging,
                             int64_t chunk_bytes, int num_blocks);

} // namespace flexkv
