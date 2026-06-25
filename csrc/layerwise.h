#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <vector>
#include <sys/eventfd.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace flexkv {

class LayerwiseTransferGroup {
public:
  LayerwiseTransferGroup(
      int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      torch::Tensor &cpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files,
      int num_layers, torch::Tensor &gpu_kv_strides_tensor,
      torch::Tensor &gpu_block_strides_tensor,
      torch::Tensor &gpu_layer_strides_tensor,
      torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size,
      const std::vector<std::vector<torch::Tensor>> &indexer_gpu_blocks = {},
      torch::Tensor indexer_cpu_blocks = torch::Tensor(),
      torch::Tensor indexer_gpu_kv_strides_tensor = torch::Tensor(),
      torch::Tensor indexer_gpu_block_strides_tensor = torch::Tensor(),
      torch::Tensor indexer_gpu_layer_strides_tensor = torch::Tensor(),
      torch::Tensor indexer_gpu_chunk_sizes_tensor = torch::Tensor(),
      std::map<int, std::vector<std::string>> indexer_ssd_files = {});

  ~LayerwiseTransferGroup();

  // Layerwise transfer: SSD->CPU + CPU->GPU
  void layerwise_transfer(
      const torch::Tensor
          &ssd_block_ids, // SSD source block ids (for disk2host)
      const torch::Tensor
          &cpu_block_ids_d2h, // CPU dest block ids (for disk2host)
      const int64_t ssd_layer_stride_in_bytes,
      const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
      const int round_robin, const int num_threads_per_device,
      const torch::Tensor
          &gpu_block_id_tensor, // GPU dest block ids (for host2device)
      const torch::Tensor
          &cpu_block_id_tensor, // CPU source block ids (for host2device)
      const int64_t cpu_kv_stride_in_bytes,
      const int64_t cpu_layer_stride_in_bytes,
      const int64_t cpu_block_stride_in_bytes,
      const int64_t cpu_chunk_size_in_bytes,
      const int64_t h2d_cpu_kv_stride_in_bytes,
      const int64_t h2d_cpu_layer_stride_in_bytes,
      const int64_t cpu_tp_stride_in_bytes, const int transfer_cta_num,
      const bool use_ce_transfer, const int num_layers,
      const int layer_granularity, const bool is_mla,
      const int counter_id = 0,
      const torch::Tensor &indexer_gpu_block_id_tensor = torch::Tensor(),
      const torch::Tensor &indexer_cpu_block_id_tensor = torch::Tensor(),
      const int64_t indexer_cpu_block_stride_in_bytes = 0,
      const int64_t indexer_cpu_layer_stride_in_bytes = 0,
      const int64_t indexer_h2d_cpu_kv_stride_in_bytes = 0,
      const int64_t indexer_h2d_cpu_layer_stride_in_bytes = 0,
      const torch::Tensor &indexer_ssd_block_ids = torch::Tensor(),
      const torch::Tensor &indexer_cpu_block_ids_d2h = torch::Tensor(),
      const int64_t indexer_ssd_layer_stride_in_bytes = 0,
      const int64_t indexer_ssd_kv_stride_in_bytes = 0,
      const int64_t indexer_cpu_chunk_size_in_bytes = 0,
      const int indexer_num_blocks_per_file = 0);

private:
  int num_gpus_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  int num_tensors_per_gpu_;
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_layer_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;

  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  std::vector<int> gpu_device_ids_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;

  // SSD IO context
  bool enable_ssd_;
  std::unique_ptr<SSDIOCTX> ioctx_;

  // Indexer fuse support
  bool enable_indexer_ = false;
  void **indexer_gpu_blocks_ = nullptr;
  void *indexer_cpu_blocks_ = nullptr;
  int indexer_num_tensors_per_gpu_ = 0;
  int64_t *indexer_gpu_kv_strides_in_bytes_ = nullptr;
  int64_t *indexer_gpu_block_strides_in_bytes_ = nullptr;
  int64_t *indexer_gpu_layer_strides_in_bytes_ = nullptr;
  int64_t *indexer_gpu_chunk_sizes_in_bytes_ = nullptr;
  BackendType indexer_backend_type_ = BackendType::SGLANG;
  std::vector<GTensorHandler> indexer_gpu_tensor_handlers_;

  // Indexer SSD IO context
  bool enable_indexer_ssd_ = false;
  std::unique_ptr<SSDIOCTX> indexer_ioctx_;

  // Layer eventfds for notification
  // Shape: [tp_size, num_counters, num_layers]
  bool enable_eventfd_;
  int tp_size_;
  int num_counters_;
  int num_layers_;
  std::vector<int> layer_eventfds_;  // Flat array
  int current_counter_id_;  // Current counter set index for this transfer

  // ===== Event-based layerwise notification (replaces cudaLaunchHostFunc) =====
  // Per-batch CUDA events recorded on each GPU stream after submitting transfers.
  // The polling thread queries these events with cudaEventQuery and writes
  // eventfds as soon as a batch completes on ALL GPUs, enabling true layerwise
  // overlap between CPU->GPU transfer and GPU compute.
  struct PollBatchInfo {
    int start_layer;
    int layers_this_batch;
    std::vector<cudaEvent_t> per_gpu_events;  // [num_gpus] one event per GPU per batch
    bool notified = false;
  };
  std::vector<PollBatchInfo> poll_batches_;
  std::atomic<bool> poll_stop_{false};
  std::thread poll_thread_;
  std::atomic<int> poll_next_batch_{0};  // Next batch index to check
  std::mutex poll_mutex_;
  std::condition_variable poll_cv_;
  bool poll_active_ = false;

  void event_polling_loop();
  void notify_layer_batch(int start_layer, int layers_this_batch);

  struct IssueWorkerState {
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::function<void()> job;
    std::exception_ptr exception;
    std::thread thread;
    bool has_job = false;
    bool done = true;
    bool stop = false;
  };

  bool use_persistent_issue_workers_ = false;
  bool issue_workers_started_ = false;
  std::vector<std::unique_ptr<IssueWorkerState>> issue_workers_;

  void start_issue_workers_if_needed();
  void stop_issue_workers();
  void submit_issue_job(int gpu_idx, std::function<void()> job);
  void wait_issue_job(int gpu_idx);
};

} // namespace flexkv
