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
#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <atomic>
#include <cstdlib>

namespace flexkv {

// [PATCH 06-10] inner timing for tp_group_transfer
// Enable: set FLEXKV_INNER_TIMING=1
// Output goes to stderr (visible in worker process log) with [INNER] prefix.
static inline bool inner_timing_enabled() {
  static const int v = []() {
    const char *e = std::getenv("FLEXKV_INNER_TIMING");
    return (e && e[0] == '1') ? 1 : 0;
  }();
  return v != 0;
}

static inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(
             steady_clock::now().time_since_epoch()).count();
}

// monotonic op id for log grouping
static std::atomic<uint64_t> g_inner_op_seq{0};

TPTransferThreadGroup::TPTransferThreadGroup(
    int num_gpus, const std::vector<int64_t> &gpu_block_ptrs_flat,
    int num_tensors_per_gpu, int64_t cpu_blocks_ptr,
    int num_layers, const std::vector<int64_t> &gpu_kv_strides_in_bytes,
    const std::vector<int64_t> &gpu_block_strides_in_bytes,
    const std::vector<int64_t> &gpu_layer_strides_in_bytes,
    const std::vector<int64_t> &gpu_chunk_sizes_in_bytes,
    const std::vector<int64_t> &gpu_device_ids) {
  num_gpus_ = num_gpus;
  num_tensors_per_gpu_ = num_tensors_per_gpu;

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];
  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = gpu_kv_strides_in_bytes[i];
    gpu_block_strides_in_bytes_[i] = gpu_block_strides_in_bytes[i];
    gpu_layer_strides_in_bytes_[i] = gpu_layer_strides_in_bytes[i];
    gpu_chunk_sizes_in_bytes_[i] = gpu_chunk_sizes_in_bytes[i];
  }

  queues_.resize(num_gpus_);
  mtxs_ = std::vector<std::mutex>(num_gpus_);
  cvs_ = std::vector<std::condition_variable>(num_gpus_);

  cudaError_t malloc_err = cudaMallocHost(
      (void **)&gpu_blocks_, num_gpus_ * num_tensors_per_gpu_ * sizeof(void *));
  if (malloc_err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                             cudaGetErrorString(malloc_err));
  }
  for (size_t i = 0; i < gpu_block_ptrs_flat.size(); ++i) {
    gpu_blocks_[i] = reinterpret_cast<void *>(gpu_block_ptrs_flat[i]);
  }

  if (num_tensors_per_gpu_ == 1) {
    backend_type_ = BackendType::TRTLLM;
  } else if (num_tensors_per_gpu_ == num_layers) {
    backend_type_ = BackendType::VLLM;
  } else if (num_tensors_per_gpu_ == num_layers * 2) {
    backend_type_ = BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported GPU block type: " +
                             std::to_string(num_tensors_per_gpu_));
  }

  gpu_tensor_handlers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    int64_t **gpu_blocks_ptr =
        reinterpret_cast<int64_t **>(gpu_blocks_ + i * num_tensors_per_gpu_);
    gpu_tensor_handlers_.emplace_back(
        backend_type_, gpu_blocks_ptr, num_layers, gpu_kv_strides_in_bytes_[i],
        gpu_block_strides_in_bytes_[i], gpu_layer_strides_in_bytes_[i]);
  }

  cpu_blocks_ = reinterpret_cast<void *>(cpu_blocks_ptr);

  gpu_device_ids_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gpu_device_ids_[i] = static_cast<int>(gpu_device_ids[i]);
  }

  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i += 1) {
    cudaError_t err = cudaSetDevice(gpu_device_ids_[i]);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(err));
    err = cudaStreamCreate(&streams_[i]);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                               cudaGetErrorString(err));
  }
  // create the thread pool
  stop_pool_ = false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = gpu_device_ids_[i];
      cudaSetDevice(device_id); // only once

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty())
            return;

          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task(); //
      }
    });
  }
}

TPTransferThreadGroup::~TPTransferThreadGroup() {
  stop_pool_ = true;
  for (auto &cv : cvs_)
    cv.notify_all();
  for (auto &t : threads_)
    if (t.joinable())
      t.join();

  cudaFreeHost(gpu_blocks_);

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

std::future<void> TPTransferThreadGroup::enqueue_for_gpu(int gpu_idx,
                                                         Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
    queues_[gpu_idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_num_cta,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const bool is_mla) {

  // [PATCH 06-10] inner timing
  const bool tracing = inner_timing_enabled();
  uint64_t op_id = tracing ? g_inner_op_seq.fetch_add(1) : 0;
  double t_enter = tracing ? now_ms() : 0.0;
  // per-thread tracing data: enqueue_to_start, kernel_launch, lambda_total
  std::vector<double> per_t_enq_to_start(num_gpus_, 0.0);
  std::vector<double> per_t_kernel_launch(num_gpus_, 0.0);
  std::vector<double> per_t_lambda_total(num_gpus_, 0.0);
  std::vector<int64_t> per_t_num_blocks(num_gpus_, 0);

  std::atomic<bool> failed{false};
  std::string error_msg;
  // threads_.clear();
  // threads_.reserve(num_gpus_);

  // Barrier sync_point(num_gpus_);
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  bool enable_sharded_d2h = is_mla && !is_host_to_device;

  double t_before_enqueue = tracing ? now_ms() : 0.0;

  for (int i = 0; i < num_gpus_; ++i) {
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      double t_lambda_start = tracing ? now_ms() : 0.0;
      try {
        int num_blocks = gpu_block_id_tensor.numel();
        if (tracing) per_t_num_blocks[i] = num_blocks;

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks = 0;
        if (enable_sharded_d2h)
          cpu_startoff_inside_chunks =
              i * gpu_chunk_sizes_in_bytes_[i] / num_gpus_;
        else if (!is_mla)
          cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
        int64_t gpu_startoff_inside_chunks =
            enable_sharded_d2h ? i * gpu_chunk_sizes_in_bytes_[i] / num_gpus_
                               : 0;
        // we assume that the chunk size is the same for all gpus,
        // even if they have different number of gpu_blocks
        int64_t chunk_size = enable_sharded_d2h
                                 ? gpu_chunk_sizes_in_bytes_[i] / num_gpus_
                                 : gpu_chunk_sizes_in_bytes_[i];

        double t_kernel_begin = tracing ? now_ms() : 0.0;

        // Dispatch to the appropriate template based on backend type
        switch (backend_type_) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
          break;
        }

        double t_kernel_end = tracing ? now_ms() : 0.0;
        if (tracing) {
          per_t_enq_to_start[i] = t_lambda_start - t_before_enqueue;
          per_t_kernel_launch[i] = t_kernel_end - t_kernel_begin;
          per_t_lambda_total[i] = t_kernel_end - t_lambda_start;
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          failed = true;
          error_msg = cudaGetErrorString(err);
        }
      } catch (const std::exception &e) {
        failed = true;
        error_msg = e.what();
      }
    }));
  }

  double t_before_wait = tracing ? now_ms() : 0.0;

  for (auto &f : futures) {
    f.get();
  }

  double t_after_wait = tracing ? now_ms() : 0.0;

  if (tracing) {
    // aggregate per-thread numbers
    double max_enq = 0, max_kernel = 0, max_lambda = 0;
    double min_enq = 1e18;
    int slow_gpu = 0;
    for (int i = 0; i < num_gpus_; ++i) {
      if (per_t_enq_to_start[i] > max_enq) max_enq = per_t_enq_to_start[i];
      if (per_t_enq_to_start[i] < min_enq) min_enq = per_t_enq_to_start[i];
      if (per_t_kernel_launch[i] > max_kernel) max_kernel = per_t_kernel_launch[i];
      if (per_t_lambda_total[i] > max_lambda) {
        max_lambda = per_t_lambda_total[i];
        slow_gpu = i;
      }
    }
    double total = t_after_wait - t_enter;
    double dispatch_overhead = t_before_enqueue - t_enter;
    double wait_after = t_after_wait - t_before_wait - max_lambda;
    fprintf(stderr,
        "[INNER] op=%lu dir=%s gpus=%d blocks=%lld total=%.1fms "
        "dispatch=%.1fms enq_min=%.1fms enq_max=%.1fms kernel_max=%.1fms "
        "lambda_max=%.1fms(gpu%d) wait_extra=%.1fms\n",
        (unsigned long)op_id,
        is_host_to_device ? "H2D" : "D2H",
        num_gpus_,
        (long long)per_t_num_blocks[0],
        total,
        dispatch_overhead,
        min_enq, max_enq, max_kernel,
        max_lambda, slow_gpu,
        wait_after);
    fflush(stderr);
  }

  if (failed) {
    throw std::runtime_error("tp_group_transfer failed: " + error_msg);
  }
}

} // namespace flexkv
