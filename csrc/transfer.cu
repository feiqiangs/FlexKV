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

#include "monitoring/metrics_manager.h"
#include "transfer.cuh"

namespace flexkv {

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
      static const int64_t segment_threshold = []() {
        const char *e = std::getenv("FLEXKV_TRANSFER_SEGMENT_THRESHOLD");
        if (e == nullptr || e[0] == '\0') return static_cast<int64_t>(8);
        int64_t v = std::atoll(e);
        return v > 0 ? v : static_cast<int64_t>(8);
      }();

      // Step 1: scan ids on host to detect contiguity / segments.
      bool src_contig = false, dst_contig = false;
      int64_t num_segments = 0;
      compute_segments(gpu_block_ids, cpu_block_ids, num_blocks, src_contig,
                       dst_contig, num_segments);

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

      // ----- Path 0: single big memcpy per (layer, kv_dim) -----
      if (chosen_path == 0) {
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
      else if (chosen_path == 1) {
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
      // ----- Path 2: gather/scatter pipeline (mirrors SGLang Path 2) -----
      //   D2H direction:
      //     - if !src_contig: GPU index_select gathers blocks into a device
      //       buffer; otherwise D2H reads the source pool directly.
      //     - if !dst_contig: D2H writes into a host pinned staging buffer
      //       and a CPU memcpy scatters segments into the final cpu_ptr;
      //       otherwise D2H writes directly to the final cpu_ptr.
      //   H2D direction:
      //     - if !src_contig: CPU memcpy gathers blocks from the cpu_ptr
      //       into a host pinned staging buffer; otherwise H2D reads the
      //       cpu_ptr directly.
      //     - if !dst_contig: H2D writes into a device buffer, then a GPU
      //       index_copy_ scatters into the final dst; otherwise H2D writes
      //       directly to the final GPU pool.
      //   Ping-pong with two CUDA events overlaps GPU work for (layer L+1,
      //   kv j) with CPU work for (layer L, kv j) when staging is needed.
      //
      // Process iterates over (layer, kv) pairs because FlexKV's kv pool is
      // {num_layers, kv_dim, num_blocks, chunk}. For MLA kv_dim==1, so the
      // loop is effectively per-layer and matches SGLang exactly.
      else {
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
        at::Tensor host_buf[2];
        if (need_dev_buf) {
          dev_buf[0] = at::empty({num_blocks, elems_per_block}, i64_cuda_opts);
          dev_buf[1] = at::empty({num_blocks, elems_per_block}, i64_cuda_opts);
        }
        if (need_host_buf) {
          auto pinned_opts = at::TensorOptions()
                                 .dtype(at::kLong)
                                 .device(at::kCPU)
                                 .pinned_memory(true);
          host_buf[0] = at::empty({num_blocks, elems_per_block}, pinned_opts);
          host_buf[1] = at::empty({num_blocks, elems_per_block}, pinned_opts);
        }

        // Ping-pong events when host staging is involved (scatter must wait
        // for the corresponding GPU stage to complete on its slot).
        cudaEvent_t pp_events[2] = {nullptr, nullptr};
        bool pp_used[2] = {false, false};
        bool need_pp_events = need_host_buf;
        if (need_pp_events) {
          cudaEventCreateWithFlags(&pp_events[0], cudaEventDisableTiming);
          cudaEventCreateWithFlags(&pp_events[1], cudaEventDisableTiming);
        }

        // CPU scatter helper for D2H non-contig dst.
        auto cpu_scatter = [&](void *staging_base, int layer_idx, int kv_idx) {
          int64_t *cpu_layer_kv_base =
              cpu_ptr_int64 +
              (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
              kv_idx * cpu_kv_stride_int64 +
              cpu_startoff_inside_chunks_int64;
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
              d2h_dst_ptr = host_buf[idx].data_ptr();
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
                cpu_scatter(host_buf[prev_idx].data_ptr(), prev_i, prev_j);
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
              cpu_gather(host_buf[idx].data_ptr(), i, j);
              h2d_src_ptr = host_buf[idx].data_ptr();
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
              cpu_scatter(host_buf[last_idx].data_ptr(), li, lj);
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

        if (need_pp_events) {
          cudaEventDestroy(pp_events[0]);
          cudaEventDestroy(pp_events[1]);
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
