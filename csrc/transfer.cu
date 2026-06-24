/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <memory>
#include <string>
#include <sys/mman.h>

#include "monitoring/metrics_manager.h"
#include "transfer.cuh"

namespace flexkv {

// [PATCH 06-10] inner-inner timing for path 1/2 inside transfer_kv_blocks
namespace {
inline bool inner_path_timing_enabled() {
  static const int v = []() {
    const char *e = std::getenv("FLEXKV_PATH_TIMING");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  return v != 0;
}
inline double now_ms_path() {
  using namespace std::chrono;
  return duration<double, std::milli>(
             steady_clock::now().time_since_epoch()).count();
}
}  // namespace



// ----------------------------------------------------------------------------
// SGLang-compatible transfer tuning helpers (copied/adapted from hicache transfer.cc)
// ----------------------------------------------------------------------------
// Env compatibility:
//   XSGL_TRANSFER_MERGED=1                -> force path2 merged path
//   XSGL_TRANSFER_SEGMENT_THRESHOLD=N     -> path1/path2 threshold, default 8
//   XSGL_TRANSFER_D2H_SEGMENT_THRESHOLD=N -> D2H-only threshold, default follows global
//   XSGL_TRANSFER_H2D_SEGMENT_THRESHOLD=N -> H2D-only threshold, default follows global
//   XSGL_TRANSFER_HUGEPAGE_MIN_BYTES=N    -> min thread-local staging buffer, default 288MB
static bool is_transfer_merged_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *env = std::getenv("XSGL_TRANSFER_MERGED");
    enabled = (env && (std::string(env) == "1" || std::string(env) == "true")) ? 1 : 0;
  }
  return enabled == 1;
}

static int64_t get_segment_count_threshold() {
  static int64_t threshold = -1;
  if (threshold < 0) {
    const char *env = std::getenv("XSGL_TRANSFER_SEGMENT_THRESHOLD");
    if (env == nullptr || env[0] == '\0') {
      // Backward compatibility during rollout: old FlexKV env may still exist.
      env = std::getenv("FLEXKV_TRANSFER_SEGMENT_THRESHOLD");
    }
    if (env) {
      threshold = std::atoll(env);
      if (threshold <= 0) threshold = 8;
    } else {
      threshold = 8;
    }
  }
  return threshold;
}

static int64_t get_directional_segment_count_threshold(bool is_host_to_device) {
  // Keep SGLang-compatible global threshold, but allow D2H/H2D split tuning.
  // This is useful on P800: D2H path1 launch loops are sensitive to compute
  // overlap, while H2D path1 is usually fine.
  static int64_t d2h_threshold = -1;
  static int64_t h2d_threshold = -1;

  int64_t &threshold = is_host_to_device ? h2d_threshold : d2h_threshold;
  if (threshold < 0) {
    const char *env = std::getenv(
        is_host_to_device ? "XSGL_TRANSFER_H2D_SEGMENT_THRESHOLD"
                          : "XSGL_TRANSFER_D2H_SEGMENT_THRESHOLD");
    if (env && env[0] != '\0') {
      threshold = std::atoll(env);
      if (threshold <= 0) threshold = get_segment_count_threshold();
    } else {
      threshold = get_segment_count_threshold();
    }
  }
  return threshold;
}

// Env gate for H2D path1 segment-level ping-pong (mirrors SGLang hicache).
// When enabled (default, can be set to 0 to disable), the H2D path1 uses a
// double-buffered hugepage staging buffer so CPU gather of segment K+1
// overlaps with H2D of segment K.
static bool is_h2d_path1_pingpong_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *env = std::getenv("FLEXKV_H2D_PATH1_PINGPONG");
    enabled = (env && std::string(env) == "0") ? 0 : 1;
  }
  return enabled == 1;
}

// Env gate for D2H path1 layer-level ping-pong (mirrors hicache MLA-CE
// transfer_kv_all_layer_mla_lf_pf_direct_ce path1).
// When enabled (default, can be set to 0 to disable), the D2H path1 uses
// a double-buffered hugepage staging buffer so D2H of layer L+1 overlaps
// with CPU scatter of layer L.
static bool is_d2h_path1_pingpong_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *env = std::getenv("FLEXKV_D2H_PATH1_PINGPONG");
    enabled = (env && std::string(env) == "0") ? 0 : 1;
  }
  return enabled == 1;
}

static size_t get_hugepage_default_min_bytes() {
  static size_t default_bytes = 0;
  if (default_bytes == 0) {
    const char *env = std::getenv("XSGL_TRANSFER_HUGEPAGE_MIN_BYTES");
    if (env && *env) {
      default_bytes = static_cast<size_t>(std::atoll(env));
    } else {
      default_bytes = static_cast<size_t>(256) * 1024 * 1152;  // 256K tokens * 1152B ~= 288MB
    }
    constexpr size_t HUGEPAGE_ALIGN = 2ULL * 1024 * 1024;
    default_bytes = (default_bytes + HUGEPAGE_ALIGN - 1) / HUGEPAGE_ALIGN * HUGEPAGE_ALIGN;
  }
  return default_bytes;
}

static void *alloc_hugepage_memory(size_t size) {
  bool is_hugepage = false;
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    fprintf(stderr, "[TRANSFER_WARN] mmap failed size=%zu errno=%d(%s)\n", size, errno, std::strerror(errno));
    return nullptr;
  }
  if (madvise(ptr, size, MADV_HUGEPAGE) == 0) {
    is_hugepage = true;
  } else {
    fprintf(stderr, "[TRANSFER_WARN] madvise(MADV_HUGEPAGE) failed ptr=%p size=%zu errno=%d(%s)\n",
            ptr, size, errno, std::strerror(errno));
  }

  // Fault-in pages up front. This avoids first-transfer page-fault spikes.
  std::memset(ptr, 0, size);

  cudaError_t err = cudaHostRegister(ptr, size, 0);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "[TRANSFER_WARN] cudaHostRegister failed ptr=%p size=%zu hugepage=%d: %s. "
            "cudaMemcpyAsync may fall back to slower pageable staging.\n",
            ptr, size, static_cast<int>(is_hugepage), cudaGetErrorString(err));
  }
  return ptr;
}

static void free_hugepage_memory(void *ptr, size_t size) {
  if (ptr) {
    cudaHostUnregister(ptr);  // best effort; ignore failure
    munmap(ptr, size);
  }
}

struct HugepageBufferHolder {
  void *ptr = nullptr;
  size_t size = 0;
  ~HugepageBufferHolder() {
    if (ptr != nullptr) {
      free_hugepage_memory(ptr, size);
      ptr = nullptr;
      size = 0;
    }
  }
};

static void *get_cached_hugepage_buffer(size_t needed_size) {
  static thread_local HugepageBufferHolder holder;
  if (holder.ptr != nullptr && holder.size >= needed_size) {
    return holder.ptr;
  }
  const size_t new_size = std::max(needed_size, get_hugepage_default_min_bytes());
  if (holder.ptr != nullptr) {
    free_hugepage_memory(holder.ptr, holder.size);
    holder.ptr = nullptr;
    holder.size = 0;
  }
  void *p = alloc_hugepage_memory(new_size);
  if (p == nullptr) return nullptr;
  holder.ptr = p;
  holder.size = new_size;
  return holder.ptr;
}

struct CudaEventPairHolder {
  cudaEvent_t events[2] = {nullptr, nullptr};
  bool initialized = false;
  int device_id = -1;
  void ensure_init(int device) {
    if (initialized) return;
    device_id = device;
    cudaSetDevice(device_id);
    cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming | cudaEventBlockingSync);
    cudaEventCreateWithFlags(&events[1], cudaEventDisableTiming | cudaEventBlockingSync);
    initialized = true;
  }
  ~CudaEventPairHolder() {
    if (initialized) {
      int prev_device = 0;
      cudaGetDevice(&prev_device);
      if (device_id >= 0) cudaSetDevice(device_id);
      if (events[0]) cudaEventDestroy(events[0]);
      if (events[1]) cudaEventDestroy(events[1]);
      if (device_id >= 0) cudaSetDevice(prev_device);
    }
  }
};

static cudaEvent_t *get_cached_pingpong_events() {
  int current_device = 0;
  cudaGetDevice(&current_device);
  static thread_local std::vector<std::unique_ptr<CudaEventPairHolder>> holders;
  if (current_device < 0) current_device = 0;
  if (static_cast<size_t>(current_device) >= holders.size()) {
    holders.resize(static_cast<size_t>(current_device) + 1);
  }
  if (!holders[current_device]) {
    holders[current_device] = std::make_unique<CudaEventPairHolder>();
  }
  holders[current_device]->ensure_init(current_device);
  return holders[current_device]->events;
}

#define FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

// ----------------------------------------------------------------------------
// CE adaptive transfer: host-side analysis utilities
// ----------------------------------------------------------------------------
//
// Scan host-side block id arrays and decide which fast-path is applicable.
//   - src_contiguous: gpu_block_ids[k+1] == gpu_block_ids[k] + 1 for all k
//   - dst_contiguous: cpu_block_ids[k+1] == cpu_block_ids[k] + 1 for all k
//   - num_segments  : number of segments after splitting on positions where
//                     either side breaks contiguity (i.e. each segment has
//                     BOTH sides simultaneously contiguous internally).
//
// We follow the same contract as SGLang's transfer_kv_all_layer_mla_ce:
// "logical id +1" implies "physical address +chunk_size", i.e. the caller is
// expected to have laid out block memory packed (block stride == chunk size)
// on both sides. No additional physical-stride check is performed here.
static inline void compute_segments(const int64_t *gpu_block_ids,
                                    const int64_t *cpu_block_ids,
                                    int num_blocks, bool &src_contiguous,
                                    bool &dst_contiguous,
                                    int64_t &num_segments) {
  src_contiguous = true;
  dst_contiguous = true;
  num_segments = (num_blocks > 0) ? 1 : 0;
  for (int k = 1; k < num_blocks; ++k) {
    bool src_step = (gpu_block_ids[k] == gpu_block_ids[k - 1] + 1);
    bool dst_step = (cpu_block_ids[k] == cpu_block_ids[k - 1] + 1);
    if (!src_step) src_contiguous = false;
    if (!dst_step) dst_contiguous = false;
    if (!src_step || !dst_step) num_segments++;
  }
}

// Templated CUDA kernel - backend type determined at compile time
template <BackendType Type>
__global__ void transfer_kv_blocks_kernel(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t copy_size, bool is_mla,
    bool is_host_to_device) {
  int kv_dim = is_mla ? 1 : 2;
  int num_chunks = num_layers * kv_dim * num_blocks;
  int64_t copy_size_in_float4 = copy_size * sizeof(int64_t) / sizeof(float4);

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;

  for (int chunk_idx = blockIdx.x * warps_per_block + warp_id;
       chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / (num_blocks * kv_dim);
    int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
    int gpu_block_idx = gpu_block_ids[chunk_idx % num_blocks];
    int cpu_block_idx = cpu_block_ids[chunk_idx % num_blocks];

    int64_t *cpu_chunk_ptr =
        cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride +
        cpu_block_idx * cpu_block_stride + cpu_startoff_inside_chunks;

    // Use template specialization to compute gpu pointer
    int64_t *gpu_ptr =
        ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
    int64_t *gpu_chunk_ptr =
        reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks;

    int64_t *src_chunk_ptr = is_host_to_device ? cpu_chunk_ptr : gpu_chunk_ptr;
    int64_t *dst_chunk_ptr = is_host_to_device ? gpu_chunk_ptr : cpu_chunk_ptr;

    for (int64_t idx = lane_id; idx < copy_size_in_float4; idx += 32) {
      float4 element;
      asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];"
                   : "=f"(element.x), "=f"(element.y), "=f"(element.z),
                     "=f"(element.w)
                   : "l"(&FLOAT4_PTR(src_chunk_ptr)[idx])
                   : "memory");
      asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" ::"l"(
                       &FLOAT4_PTR(dst_chunk_ptr)[idx]),
                   "f"(element.x), "f"(element.y), "f"(element.z),
                   "f"(element.w)
                   : "memory");
    }
  }
}

// Templated host function
template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, void *cpu_ptr, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t cpu_startoff_inside_chunks, int64_t chunk_size_in_bytes,
    cudaStream_t stream, int transfer_num_cta, bool is_host_to_device,
    bool use_ce_transfer, bool is_mla, bool sync) {

  int block_size = 1024;

  int block_count = transfer_num_cta;

  int64_t *cpu_ptr_int64 = reinterpret_cast<int64_t *>(cpu_ptr);
  int64_t cpu_kv_stride_int64 = cpu_kv_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_block_stride_int64 = cpu_block_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_layer_stride_int64 = cpu_layer_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_startoff_inside_chunks_int64 =
      cpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t gpu_startoff_inside_chunks_int64 =
      gpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t chunk_size_in_int64 = chunk_size_in_bytes / sizeof(int64_t);

  dim3 blockDim(block_size);
  dim3 gridDim(block_count);

  // CE transfer mode (Copy Engine using cudaMemcpyAsync)
  if (use_ce_transfer) {
    int kv_dim = is_mla ? 1 : 2;

    // Adaptive multi-path CE transfer is enabled only when is_mla == true.
    // For the non-MLA case (kv_dim == 2) we keep the legacy per-block loop.
    if (is_mla) {
      // Read tunables once (cached as static).
      static const int force_path_int = []() {
        const char *e = std::getenv("FLEXKV_TRANSFER_FORCE_PATH");
        if (e == nullptr || e[0] == '\0') return -1;
        int v = std::atoi(e);
        return (v == 0 || v == 1 || v == 2) ? v : -1;
      }();
      const int64_t segment_threshold = get_directional_segment_count_threshold(is_host_to_device);

      // Step 1: scan ids on host to detect contiguity / segments.
      // compute_segments sets:
      //   src_contig  = gpu_block_ids contiguous (GPU-side)
      //   dst_contig  = cpu_block_ids contiguous (CPU-side)
      // This naming is correct for D2H (GPU=src, CPU=dst), but for H2D the
      // actual data source is CPU and destination is GPU, so we swap the two
      // flags for H2D to keep the rest of the code semantically consistent.
      bool src_contig = false, dst_contig = false;
      int64_t num_segments = 0;
      compute_segments(gpu_block_ids, cpu_block_ids, num_blocks, src_contig,
                       dst_contig, num_segments);
      if (is_host_to_device) {
        std::swap(src_contig, dst_contig);
      }

      // Step 2: choose path (mirrors SGLang transfer_kv_all_layer_mla_ce).
      //   Path 0: BOTH sides logically contiguous -> single memcpy per
      //           layer x kv_dim.
      //   Path 1: num_segments <= threshold -> per-segment memcpy.
      //   Path 2: otherwise -> gather/scatter pipeline.
      int chosen_path = -1;
      if (force_path_int == 0) chosen_path = 0;
      else if (force_path_int == 1) chosen_path = 1;
      else if (force_path_int == 2) chosen_path = 2;
      else if (src_contig && dst_contig) chosen_path = 0;
      else if (num_segments <= segment_threshold) chosen_path = 1;
      else chosen_path = 2;
      if (is_transfer_merged_enabled()) {
        // Match SGLang XSGL_TRANSFER_MERGED behavior: force merged path for testing/optimization.
        chosen_path = 2;
      }
      // When block_stride != chunk_size (e.g. sharded D2H where chunk_size =
      // gpu_chunk/num_gpus but block_stride = gpu_chunk), Path 0's single big
      // memcpy crosses block boundaries and corrupts data. Redirect to Path 1
      // which correctly uses per-segment offsets. Path 2 is only needed when
      // there are too many segments.
      if (cpu_block_stride_in_bytes != chunk_size_in_bytes && chosen_path == 0) {
        chosen_path = 1;
      }

      // ----- Path 0: single big memcpy per (layer, kv_dim) -----
      // Only valid when blocks are physically contiguous in CPU memory
      // (block_stride == chunk_size). When block_stride > chunk_size (e.g.
      // sharded D2H where chunk_size = gpu_chunk/num_gpus but block_stride =
      // gpu_chunk), a single big memcpy would write across block boundaries
      // and corrupt data.
      if (chosen_path == 0 &&
          cpu_block_stride_in_bytes == chunk_size_in_bytes) {
        int64_t big_size = chunk_size_in_bytes * num_blocks;
        for (int i = 0; i < num_layers; i++) {
          for (int j = 0; j < kv_dim; j++) {
            int64_t *cpu_chunk_ptr =
                cpu_ptr_int64 +
                (i + start_layer_id) * cpu_layer_stride_int64 +
                j * cpu_kv_stride_int64 +
                cpu_block_ids[0] * cpu_block_stride_int64 +
                cpu_startoff_inside_chunks_int64;
            int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                            i + start_layer_id, j,
                                            gpu_block_ids[0]);
            int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                     gpu_startoff_inside_chunks_int64;
            if (is_host_to_device) {
              cudaMemcpyAsync(gpu_chunk_ptr, cpu_chunk_ptr, big_size,
                              cudaMemcpyHostToDevice, stream);
            } else {
              cudaMemcpyAsync(cpu_chunk_ptr, gpu_chunk_ptr, big_size,
                              cudaMemcpyDeviceToHost, stream);
            }
            FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, big_size);
          }
        }
      }
      // ----- Path 1: per-segment memcpy -----
      else if (chosen_path == 1 || chosen_path == 0) {
        // [PATCH 06-10] path 1 inner timing
        const bool _ptiming = inner_path_timing_enabled();
        double _t0 = _ptiming ? now_ms_path() : 0.0;

        // Build segment list on host: each segment is (start_k, run_len)
        // where for [start_k, start_k+run_len) BOTH gpu and cpu ids step by 1.
        // Mirrors SGLang Path 1: pure logical-id check, no per-segment GPU
        // physical-contiguity verification (the caller is contracted to lay
        // out memory packed).
        std::vector<std::pair<int, int>> segments;
        segments.reserve(static_cast<size_t>(num_segments));
        int seg_start = 0;
        for (int k = 1; k <= num_blocks; ++k) {
          bool break_here = (k == num_blocks);
          if (!break_here) {
            bool src_step = (gpu_block_ids[k] == gpu_block_ids[k - 1] + 1);
            bool dst_step = (cpu_block_ids[k] == cpu_block_ids[k - 1] + 1);
            if (!src_step || !dst_step) break_here = true;
          }
          if (break_here) {
            segments.emplace_back(seg_start, k - seg_start);
            seg_start = k;
          }
        }

        double _t_seg_done = _ptiming ? now_ms_path() : 0.0;
        size_t _seg_count = segments.size();

        // Ping-pong is only beneficial in batch mode (num_layers > 1) where
        // layer-to-layer overlap is possible. In layerwise mode (num_layers
        // == 1), ping-pong adds cudaEventSynchronize overhead with zero
        // overlap benefit, and the local event_used reset across calls can
        // cause staging buffer corruption when the previous layer's H2D DMA
        // is still reading the thread-local cached host_bufs.
        if (is_host_to_device && is_h2d_path1_pingpong_enabled() && num_layers > 1) {
          // ---- H2D path1 with segment-level ping-pong (mirrors SGLang hicache) ----
          // Allocate double-buffered hugepage staging so CPU gather of seg K+1
          // overlaps with H2D of seg K. Each segment is gathered from cpu_ptr
          // into a staging slot, then one cudaMemcpyAsync H2D issues; while
          // the GPU DMA runs, the CPU gathers the next segment into the other
          // slot. When a slot is needed again, we wait for its H2D to finish.
          const size_t max_seg_bytes =
              static_cast<size_t>(num_blocks) * static_cast<size_t>(chunk_size_in_bytes);
          void *host_base = get_cached_hugepage_buffer(max_seg_bytes * 2);
          TORCH_CHECK(host_base != nullptr,
                      "path1 H2D staging buffer allocation failed");
          void *host_bufs[2] = {
              host_base,
              static_cast<char *>(host_base) + max_seg_bytes,
          };
          cudaEvent_t *pp_events = get_cached_pingpong_events();
          bool event_used[2] = {false, false};

          for (int i = 0; i < num_layers; i++) {
            for (int j = 0; j < kv_dim; j++) {
              int seg_idx = 0;
              for (auto &seg : segments) {
                int sk = seg.first;
                int run = seg.second;
                int64_t seg_size = chunk_size_in_bytes * run;
                const int idx = seg_idx & 1;

                // Wait for previous use of this buffer slot to finish.
                if (event_used[idx]) {
                  cudaEventSynchronize(pp_events[idx]);
                }

                // CPU gather: copy from cpu blocks into staging buffer.
                int64_t *cpu_chunk_ptr =
                    cpu_ptr_int64 +
                    (i + start_layer_id) * cpu_layer_stride_int64 +
                    j * cpu_kv_stride_int64 +
                    cpu_block_ids[sk] * cpu_block_stride_int64 +
                    cpu_startoff_inside_chunks_int64;
                std::memcpy(host_bufs[idx], cpu_chunk_ptr,
                            static_cast<size_t>(seg_size));

                // H2D from staging buffer to GPU.
                int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                                i + start_layer_id, j,
                                                gpu_block_ids[sk]);
                int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                         gpu_startoff_inside_chunks_int64;
                cudaMemcpyAsync(gpu_chunk_ptr, host_bufs[idx], seg_size,
                                cudaMemcpyHostToDevice, stream);
                cudaEventRecord(pp_events[idx], stream);
                event_used[idx] = true;
                FLEXKV_GPU_CPU_TRANSFER(true, seg_size);
                seg_idx++;
              }
            }
          }
          // Flush remaining in-flight H2D segments.
          if (event_used[0]) cudaEventSynchronize(pp_events[0]);
          if (event_used[1]) cudaEventSynchronize(pp_events[1]);
        } else if (!is_host_to_device && is_d2h_path1_pingpong_enabled() && num_layers > 1) {
          // -- D2H path1 with layer-level ping-pong (mirrors hicache MLA-CE
          //    transfer_kv_all_layer_mla_lf_pf_direct_ce path1) --
          // Double-buffered hugepage staging: D2H of layer L+1 overlaps with
          // CPU scatter of layer L.  All segments for one (layer,kv) are
          // D2H-ed into a contiguous staging slot, then while the GPU
          // processes the next slot the CPU scatters the previous one into
          // the final cpu_ptr positions.
          const size_t layer_buf_bytes =
              static_cast<size_t>(num_blocks) * static_cast<size_t>(chunk_size_in_bytes);
          void *host_base = get_cached_hugepage_buffer(layer_buf_bytes * 2);
          TORCH_CHECK(host_base != nullptr,
                      "path1 D2H staging buffer allocation failed");
          void *host_bufs[2] = {
              host_base,
              static_cast<char *>(host_base) + layer_buf_bytes,
          };
          cudaEvent_t *pp_events = get_cached_pingpong_events();

          const int64_t total_iters =
              static_cast<int64_t>(num_layers) * static_cast<int64_t>(kv_dim);
          for (int64_t it = 0; it < total_iters; ++it) {
            const int layer_i = static_cast<int>(it / kv_dim);
            const int kv_j   = static_cast<int>(it % kv_dim);
            const int idx    = static_cast<int>(it & 1);
            void *buf = host_bufs[idx];

            // D2H all segments for this (layer,kv) into staging buf.
            int64_t seg_offset = 0;
            for (auto &seg : segments) {
              int sk = seg.first;
              int run = seg.second;
              int64_t seg_size = chunk_size_in_bytes * run;
              int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                              layer_i + start_layer_id, kv_j,
                                              gpu_block_ids[sk]);
              int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                       gpu_startoff_inside_chunks_int64;
              char *staging_dst = static_cast<char *>(buf) + seg_offset;
              cudaMemcpyAsync(staging_dst, gpu_chunk_ptr, seg_size,
                              cudaMemcpyDeviceToHost, stream);
              seg_offset += seg_size;
              FLEXKV_GPU_CPU_TRANSFER(false, seg_size);
            }
            cudaEventRecord(pp_events[idx], stream);

            // Overlap: CPU scatter the PREVIOUS (layer,kv) while GPU
            // executes D2H for the current one.
            if (it >= 1) {
              const int prev_idx = static_cast<int>((it - 1) & 1);
              cudaEventSynchronize(pp_events[prev_idx]);
              const int prev_layer = static_cast<int>((it - 1) / kv_dim);
              const int prev_kv    = static_cast<int>((it - 1) % kv_dim);
              int64_t scatter_off = 0;
              for (auto &seg : segments) {
                int sk = seg.first;
                int run = seg.second;
                int64_t seg_size = chunk_size_in_bytes * run;
                int64_t *cpu_dst =
                    cpu_ptr_int64 +
                    (prev_layer + start_layer_id) * cpu_layer_stride_int64 +
                    prev_kv * cpu_kv_stride_int64 +
                    cpu_block_ids[sk] * cpu_block_stride_int64 +
                    cpu_startoff_inside_chunks_int64;
                std::memcpy(cpu_dst,
                            static_cast<char *>(host_bufs[prev_idx]) + scatter_off,
                            static_cast<size_t>(seg_size));
                scatter_off += seg_size;
              }
            }
          }
          // Flush the final (layer,kv).
          if (total_iters >= 1) {
            const int last_idx = static_cast<int>((total_iters - 1) & 1);
            cudaEventSynchronize(pp_events[last_idx]);
            const int last_layer = static_cast<int>((total_iters - 1) / kv_dim);
            const int last_kv    = static_cast<int>((total_iters - 1) % kv_dim);
            int64_t scatter_off = 0;
            for (auto &seg : segments) {
              int sk = seg.first;
              int run = seg.second;
              int64_t seg_size = chunk_size_in_bytes * run;
              int64_t *cpu_dst =
                  cpu_ptr_int64 +
                  (last_layer + start_layer_id) * cpu_layer_stride_int64 +
                  last_kv * cpu_kv_stride_int64 +
                  cpu_block_ids[sk] * cpu_block_stride_int64 +
                  cpu_startoff_inside_chunks_int64;
              std::memcpy(cpu_dst,
                          static_cast<char *>(host_bufs[last_idx]) + scatter_off,
                          static_cast<size_t>(seg_size));
              scatter_off += seg_size;
            }
          }
        } else {
          for (int i = 0; i < num_layers; i++) {
            for (int j = 0; j < kv_dim; j++) {
              for (auto &seg : segments) {
                int sk = seg.first;
                int run = seg.second;
                int64_t seg_size = chunk_size_in_bytes * run;
                int64_t *cpu_chunk_ptr =
                    cpu_ptr_int64 +
                    (i + start_layer_id) * cpu_layer_stride_int64 +
                    j * cpu_kv_stride_int64 +
                    cpu_block_ids[sk] * cpu_block_stride_int64 +
                    cpu_startoff_inside_chunks_int64;
                int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                                i + start_layer_id, j,
                                                gpu_block_ids[sk]);
                int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                         gpu_startoff_inside_chunks_int64;
                if (is_host_to_device) {
                  cudaMemcpyAsync(gpu_chunk_ptr, cpu_chunk_ptr, seg_size,
                                  cudaMemcpyHostToDevice, stream);
                } else {
                  cudaMemcpyAsync(cpu_chunk_ptr, gpu_chunk_ptr, seg_size,
                                  cudaMemcpyDeviceToHost, stream);
                }
                FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, seg_size);
              }
            }
          }
        }
        if (_ptiming) {
          double _t_launch_done = now_ms_path();
          int64_t total_launches = (int64_t)num_layers * (int64_t)kv_dim *
                                    (int64_t)_seg_count;
          fprintf(stderr,
              "[PATH1] dir=%s blocks=%d layers=%d kv_dim=%d segs=%zu "
              "total_launches=%lld build_seg=%.2fms launch_loop=%.2fms\n",
              is_host_to_device ? "H2D" : "D2H",
              num_blocks, num_layers, kv_dim, _seg_count,
              (long long)total_launches,
              _t_seg_done - _t0,
              _t_launch_done - _t_seg_done);
          fflush(stderr);
        }
      }
      // ----- Path 2: gather/scatter pipeline (mirrors SGLang Path 2) -----
      //   When block_stride != chunk_size (e.g. sharded D2H), we force
      //   non-contiguous treatment so per-block scatter/gather is used.
      //   ...
      else {
        // When block_stride != chunk_size, force CPU-side non-contig so the
        // per-block scatter/gather (which correctly uses block_stride) is used.
        // GPU-side contiguity is preserved for the D2H/H2D staging transfer.
        if (cpu_block_stride_in_bytes != chunk_size_in_bytes) {
          if (is_host_to_device) {
            src_contig = false;  // H2D: CPU src needs per-block gather
          } else {
            dst_contig = false;  // D2H: CPU dst needs per-block scatter
          }
        }
        // ---- Pre-compute helpers -----------------------------------------
        // chunk size measured in int64 elements (chunk size is guaranteed
        // multiple of 8 because callers built it from float4-aligned strides).
        TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
                    "Path 2 requires chunk_size_in_bytes %% 8 == 0");
        const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
        const int64_t buffer_size_bytes =
            static_cast<int64_t>(num_blocks) * chunk_size_in_bytes;

        // Bind ATen to our cuda stream so index_select_out / index_copy_
        // run on the same stream as cudaMemcpyAsync. Without this the ATen
        // ops execute on the default current stream and may race with our
        // pipeline (H2D into dev_buf followed by index_copy_ scatter).
        int cur_dev = 0;
        cudaGetDevice(&cur_dev);
        c10::cuda::CUDAStream aten_stream =
            c10::cuda::getStreamFromExternal(stream, cur_dev);
        c10::cuda::CUDAStreamGuard stream_guard(aten_stream);

        // Find max gpu / cpu indices to size at::from_blob views.
        int64_t max_gpu_id = 0, max_cpu_id = 0;
        for (int k = 0; k < num_blocks; ++k) {
          if (gpu_block_ids[k] > max_gpu_id) max_gpu_id = gpu_block_ids[k];
          if (cpu_block_ids[k] > max_cpu_id) max_cpu_id = cpu_block_ids[k];
        }

        // Wrap gpu_block_ids / cpu_block_ids (host int64*) as cuda tensors
        // for index_select / index_copy_.
        auto i64_cuda_opts =
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
        auto i64_cpu_opts =
            at::TensorOptions().dtype(at::kLong).device(at::kCPU);
        at::Tensor gpu_ids_cpu = at::from_blob(
            const_cast<int64_t *>(gpu_block_ids), {num_blocks}, i64_cpu_opts);
        at::Tensor gpu_ids_cuda =
            (!src_contig) ? gpu_ids_cpu.to(at::kCUDA, /*non_blocking=*/true)
                          : at::Tensor();
        at::Tensor cpu_ids_cpu = at::from_blob(
            const_cast<int64_t *>(cpu_block_ids), {num_blocks}, i64_cpu_opts);
        at::Tensor dst_ids_cuda =
            (is_host_to_device && !dst_contig)
                ? gpu_ids_cpu.to(at::kCUDA, /*non_blocking=*/true)
                : at::Tensor();
        // For D2H scatter we need host-side cpu_block_ids array (already host).

        // ---- Allocate ping-pong buffers ----------------------------------
        // Device buffers: needed for D2H when src is non-contig (gather
        // target) or for H2D when dst is non-contig (scatter source).
        bool need_dev_buf =
            (!is_host_to_device && !src_contig) ||
            (is_host_to_device && !dst_contig);
        // Host pinned buffers: needed for D2H when dst is non-contig (scatter
        // staging) or for H2D when src is non-contig (gather staging).
        bool need_host_buf =
            (!is_host_to_device && !dst_contig) ||
            (is_host_to_device && !src_contig);

        at::Tensor dev_buf[2];
        if (need_dev_buf) {
          dev_buf[0] = at::empty({num_blocks, elems_per_block}, i64_cuda_opts);
          dev_buf[1] = at::empty({num_blocks, elems_per_block}, i64_cuda_opts);
        }

        // SGLang-style thread-local hugepage pinned staging buffer.
        // Avoid per-transfer at::empty(... pinned_memory=true) / cudaHostAlloc overhead.
        void *host_buf[2] = {nullptr, nullptr};
        if (need_host_buf) {
          void *host_buffer_base = get_cached_hugepage_buffer(static_cast<size_t>(buffer_size_bytes) * 2);
          TORCH_CHECK(host_buffer_base != nullptr, "Failed to allocate cached hugepage host buffer for FlexKV transfer");
          host_buf[0] = host_buffer_base;
          host_buf[1] = static_cast<char *>(host_buffer_base) + buffer_size_bytes;
        }

        // SGLang-style thread-local ping-pong events when host staging is involved.
        cudaEvent_t *pp_events = need_host_buf ? get_cached_pingpong_events() : nullptr;
        bool pp_used[2] = {false, false};
        bool need_pp_events = need_host_buf;

        // CPU scatter helper for D2H non-contig dst.
        auto cpu_scatter = [&](void *staging_base, int layer_idx, int kv_idx) {
          int64_t *cpu_layer_kv_base =
              cpu_ptr_int64 +
              (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
              kv_idx * cpu_kv_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          // When block_stride == chunk_size, blocks are physically contiguous
          // and we can do per-segment bulk memcpy. When block_stride !=
          // chunk_size (e.g. sharded D2H), each block must be copied
          // individually to block_ids[k] * block_stride.
          if (cpu_block_stride_in_bytes == chunk_size_in_bytes) {
            int sb = 0;
            while (sb < num_blocks) {
              int se = sb + 1;
              while (se < num_blocks &&
                     cpu_block_ids[se] == cpu_block_ids[se - 1] + 1) {
                ++se;
              }
              const int run = se - sb;
              const char *src =
                  static_cast<const char *>(staging_base) +
                  static_cast<int64_t>(sb) * chunk_size_in_bytes;
              int64_t *dst =
                  cpu_layer_kv_base +
                  cpu_block_ids[sb] * cpu_block_stride_int64;
              std::memcpy(dst, src, static_cast<size_t>(run) * chunk_size_in_bytes);
              sb = se;
            }
          } else {
            for (int k = 0; k < num_blocks; ++k) {
              const char *src =
                  static_cast<const char *>(staging_base) +
                  static_cast<int64_t>(k) * chunk_size_in_bytes;
              int64_t *dst =
                  cpu_layer_kv_base +
                  cpu_block_ids[k] * cpu_block_stride_int64;
              std::memcpy(dst, src, static_cast<size_t>(chunk_size_in_bytes));
            }
          }
        };

        // CPU gather helper for H2D non-contig src.
        auto cpu_gather = [&](void *staging_base, int layer_idx, int kv_idx) {
          int64_t *cpu_layer_kv_base =
              cpu_ptr_int64 +
              (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
              kv_idx * cpu_kv_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          for (int k = 0; k < num_blocks; ++k) {
            const int64_t *src =
                cpu_layer_kv_base +
                cpu_block_ids[k] * cpu_block_stride_int64;
            char *dst = static_cast<char *>(staging_base) +
                        static_cast<int64_t>(k) * chunk_size_in_bytes;
            std::memcpy(dst, src, static_cast<size_t>(chunk_size_in_bytes));
          }
        };

        const int64_t total_iters =
            static_cast<int64_t>(num_layers) * static_cast<int64_t>(kv_dim);
        for (int64_t it = 0; it < total_iters; ++it) {
          const int i = static_cast<int>(it / kv_dim);
          const int j = static_cast<int>(it % kv_dim);
          const int idx = static_cast<int>(it & 1);
          const int prev_idx = idx ^ 1;

          // ---- Per-(layer, kv) GPU pool view (for index_select / scatter)
          // GPU base ptr at (layer i+start, kv j, block 0) viewed as int64.
          int64_t *gpu_layer_kv_base =
              ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
          int64_t *gpu_chunk_ptr_offsetted =
              gpu_layer_kv_base + gpu_startoff_inside_chunks_int64;
          (void)gpu_chunk_ptr_offsetted;

          if (!is_host_to_device) {
            // ============= D2H ===========================================
            // Step A (GPU gather, when !src_contig): index_select_out into
            // dev_buf[idx].
            const int64_t *d2h_src_ptr_int64 = nullptr;
            if (src_contig) {
              // Direct contiguous source slice starting at gpu_block_ids[0].
              d2h_src_ptr_int64 =
                  ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j,
                               gpu_block_ids[0]);
            } else {
              at::Tensor src_view = at::from_blob(
                  gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
                  i64_cuda_opts);
              at::index_select_out(dev_buf[idx], src_view, /*dim=*/0,
                                   gpu_ids_cuda);
              d2h_src_ptr_int64 =
                  reinterpret_cast<int64_t *>(dev_buf[idx].data_ptr());
            }

            // Step B (D2H copy)
            void *d2h_dst_ptr;
            if (dst_contig) {
              // Direct write to final cpu position (no scatter needed).
              d2h_dst_ptr =
                  cpu_ptr_int64 +
                  (i + start_layer_id) * cpu_layer_stride_int64 +
                  j * cpu_kv_stride_int64 +
                  cpu_block_ids[0] * cpu_block_stride_int64 +
                  cpu_startoff_inside_chunks_int64;
            } else {
              d2h_dst_ptr = host_buf[idx];
            }
            cudaMemcpyAsync(d2h_dst_ptr, d2h_src_ptr_int64, buffer_size_bytes,
                            cudaMemcpyDeviceToHost, stream);
            FLEXKV_GPU_CPU_TRANSFER(false, buffer_size_bytes);

            if (need_pp_events) {
              cudaEventRecord(pp_events[idx], stream);
              pp_used[idx] = true;
              // Step C (CPU scatter for the *previous* slot).
              if (it >= 1 && pp_used[prev_idx]) {
                cudaEventSynchronize(pp_events[prev_idx]);
                const int prev_i = static_cast<int>((it - 1) / kv_dim);
                const int prev_j = static_cast<int>((it - 1) % kv_dim);
                cpu_scatter(host_buf[prev_idx], prev_i, prev_j);
              }
            }
          } else {
            // ============= H2D ===========================================
            // Step A (CPU gather, when !src_contig): memcpy into host_buf[idx].
            const void *h2d_src_ptr;
            if (src_contig) {
              // Direct contiguous source slice in cpu_ptr.
              h2d_src_ptr =
                  cpu_ptr_int64 +
                  (i + start_layer_id) * cpu_layer_stride_int64 +
                  j * cpu_kv_stride_int64 +
                  cpu_block_ids[0] * cpu_block_stride_int64 +
                  cpu_startoff_inside_chunks_int64;
            } else {
              if (need_pp_events && pp_used[idx]) {
                // Make sure previous use of host_buf[idx] (the H2D that
                // consumed it) has completed before we overwrite it.
                cudaEventSynchronize(pp_events[idx]);
              }
              cpu_gather(host_buf[idx], i, j);
              h2d_src_ptr = host_buf[idx];
            }

            // Step B (H2D copy)
            void *h2d_dst_ptr;
            if (dst_contig) {
              // Direct write to the final GPU position.
              h2d_dst_ptr = ptr_at<Type>(gpu_tensor_handler,
                                         i + start_layer_id, j,
                                         gpu_block_ids[0]);
            } else {
              h2d_dst_ptr = dev_buf[idx].data_ptr();
            }
            cudaMemcpyAsync(h2d_dst_ptr, h2d_src_ptr, buffer_size_bytes,
                            cudaMemcpyHostToDevice, stream);
            FLEXKV_GPU_CPU_TRANSFER(true, buffer_size_bytes);

            if (need_pp_events) {
              cudaEventRecord(pp_events[idx], stream);
              pp_used[idx] = true;
            }

            // Step C (GPU scatter, when !dst_contig)
            if (!dst_contig) {
              at::Tensor dst_view = at::from_blob(
                  gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
                  i64_cuda_opts);
              dst_view.index_copy_(/*dim=*/0, dst_ids_cuda, dev_buf[idx]);
            }
          }
        }

        // ---- Drain pending CPU scatter (D2H non-contig dst) ---------------
        if (!is_host_to_device && need_pp_events) {
          // Last iteration's slot still needs scatter.
          const int64_t last = total_iters - 1;
          if (last >= 0) {
            const int last_idx = static_cast<int>(last & 1);
            if (pp_used[last_idx]) {
              cudaEventSynchronize(pp_events[last_idx]);
              const int li = static_cast<int>(last / kv_dim);
              const int lj = static_cast<int>(last % kv_dim);
              cpu_scatter(host_buf[last_idx], li, lj);
            }
          }
        }

        // For H2D direct (dst_contig) or H2D with GPU scatter, ensure
        // GPU-side work is observable to caller before return (the outer
        // `if (sync)` only fires when caller asked; we additionally sync
        // here when GPU scatter ran on non-default stream paths).
        if (is_host_to_device && !dst_contig) {
          cudaStreamSynchronize(stream);
        }
      }
    } else {
      // Legacy non-MLA path: keep the original per-block loop unchanged.
      for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < kv_dim; j++) {
          for (int k = 0; k < num_blocks; k++) {
            int64_t gpu_block_idx = gpu_block_ids[k];
            int64_t cpu_block_idx = cpu_block_ids[k];

            int64_t *cpu_chunk_ptr =
                cpu_ptr_int64 +
                (i + start_layer_id) * cpu_layer_stride_int64 +
                j * cpu_kv_stride_int64 +
                cpu_block_idx * cpu_block_stride_int64 +
                cpu_startoff_inside_chunks_int64;

            int64_t *gpu_ptr = ptr_at<Type>(
                gpu_tensor_handler, i + start_layer_id, j, gpu_block_idx);
            int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                     gpu_startoff_inside_chunks_int64;

            if (is_host_to_device) {
              cudaMemcpyAsync(gpu_chunk_ptr, cpu_chunk_ptr, chunk_size_in_bytes,
                              cudaMemcpyHostToDevice, stream);
            } else {
              cudaMemcpyAsync(cpu_chunk_ptr, gpu_chunk_ptr, chunk_size_in_bytes,
                              cudaMemcpyDeviceToHost, stream);
            }
            // Record transfer metrics after each cudaMemcpyAsync submission
            // Direction convention (from GPU perspective):
            //   - is_host_to_device=true  -> read (CPU->GPU, data flows INTO GPU)
            //   - is_host_to_device=false -> write (GPU->CPU, data flows OUT of GPU)
            FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes);
          }
        }
      }
    }
  } else {
    // Custom kernel transfer
    transfer_kv_blocks_kernel<Type><<<gridDim, blockDim, 0, stream>>>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_int64, is_mla, is_host_to_device);

    // Record transfer metrics after kernel launch (cannot record inside kernel)
    // Total bytes = actual_chunk_bytes * num_layers * kv_dim * num_blocks
    // Note: Kernel transfers in float4 units, so we calculate aligned bytes to
    // match Direction convention (from GPU perspective):
    //   - is_host_to_device=true  -> read (CPU->GPU, data flows INTO GPU)
    //   - is_host_to_device=false -> write (GPU->CPU, data flows OUT of GPU)
    int kv_dim = is_mla ? 1 : 2;
    // Calculate actual bytes transferred (aligned to float4, matching kernel
    // logic)
    int64_t actual_chunk_bytes =
        (chunk_size_in_int64 * sizeof(int64_t) / sizeof(float4)) *
        sizeof(float4);
    FLEXKV_GPU_CPU_TRANSFER(
        is_host_to_device,
        actual_chunk_bytes * static_cast<int64_t>(num_layers) *
            static_cast<int64_t>(kv_dim) * static_cast<int64_t>(num_blocks));
  }
  if (sync) {
    cudaStreamSynchronize(stream);
  }
}

// Explicit template instantiations
template void transfer_kv_blocks<BackendType::VLLM>(int, int, int, int64_t *,
                                                    GTensorHandler, int64_t,
                                                    int64_t *, void *, int64_t,
                                                    int64_t, int64_t, int64_t,
                                                    int64_t, cudaStream_t, int,
                                                    bool, bool, bool, bool);

template void transfer_kv_blocks<BackendType::TRTLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool);

template void transfer_kv_blocks<BackendType::SGLANG>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool);

} // namespace flexkv
