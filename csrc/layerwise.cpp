#include "layerwise.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

namespace flexkv {

// ===== Event-based layerwise notification =====
// Replaces cudaLaunchHostFunc (which does not work on Hygon DCU).
// The polling thread queries per-batch CUDA events and writes eventfds
// as soon as each batch completes on all GPUs.

void LayerwiseTransferGroup::notify_layer_batch(int start_layer,
                                                 int layers_this_batch) {
  if (!enable_eventfd_ || layer_eventfds_.empty()) return;

  int offset = current_counter_id_ * tp_size_ * num_layers_;
  int *eventfds_base = layer_eventfds_.data() + offset;

  for (int layer = start_layer;
       layer < start_layer + layers_this_batch; ++layer) {
    for (int tp_rank = 0; tp_rank < tp_size_; ++tp_rank) {
      int fd = eventfds_base[tp_rank * num_layers_ + layer];
      if (fd >= 0) {
        uint64_t val = 2;  // Support both get_key_buffer and get_value_buffer waits
        ssize_t ret = write(fd, &val, sizeof(val));
        (void)ret;
      }
    }
  }
  fprintf(stderr, "[LW-EVT] notified layers [%d, %d) counter_id=%d\n",
          start_layer, start_layer + layers_this_batch, current_counter_id_);
}

void LayerwiseTransferGroup::event_polling_loop() {
  fprintf(stderr, "[LW-EVT] polling thread started, num_batches=%zu\n",
          poll_batches_.size());

  while (!poll_stop_.load(std::memory_order_acquire)) {
    int next = poll_next_batch_.load(std::memory_order_acquire);
    if (next >= (int)poll_batches_.size()) {
      // All batches notified, exit
      break;
    }

    PollBatchInfo &batch = poll_batches_[next];
    if (batch.notified) {
      poll_next_batch_.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }

    // Check if all GPUs have completed this batch
    bool all_done = true;
    for (int g = 0; g < num_gpus_ && all_done; ++g) {
      cudaSetDevice(gpu_device_ids_[g]);
      cudaError_t err = cudaEventQuery(batch.per_gpu_events[g]);
      if (err == cudaErrorNotReady) {
        all_done = false;
      } else if (err != cudaSuccess) {
        fprintf(stderr, "[LW-EVT] cudaEventQuery error GPU=%d batch=%d: %s\n",
                g, next, cudaGetErrorString(err));
        all_done = false;
      }
    }

    if (all_done) {
      batch.notified = true;
      notify_layer_batch(batch.start_layer, batch.layers_this_batch);
      poll_next_batch_.fetch_add(1, std::memory_order_acq_rel);
    } else {
      // Brief yield to avoid burning CPU
      std::this_thread::yield();
    }
  }

  // Safety: if we exit early but some batches weren't notified, notify them now
  for (auto &batch : poll_batches_) {
    if (!batch.notified) {
      // Force-check with cudaEventSynchronize (blocking) as fallback
      for (int g = 0; g < num_gpus_; ++g) {
        cudaSetDevice(gpu_device_ids_[g]);
        cudaEventSynchronize(batch.per_gpu_events[g]);
      }
      batch.notified = true;
      notify_layer_batch(batch.start_layer, batch.layers_this_batch);
    }
  }

  fprintf(stderr, "[LW-EVT] polling thread exiting, all batches notified\n");
}

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files,
    int num_layers, torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_layer_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
    int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size,
    const std::vector<std::vector<torch::Tensor>> &indexer_gpu_blocks,
    torch::Tensor indexer_cpu_blocks,
    torch::Tensor indexer_gpu_kv_strides_tensor,
    torch::Tensor indexer_gpu_block_strides_tensor,
    torch::Tensor indexer_gpu_layer_strides_tensor,
    torch::Tensor indexer_gpu_chunk_sizes_tensor,
    std::map<int, std::vector<std::string>> indexer_ssd_files) {

  num_gpus_ = num_gpus;
  num_layers_ = num_layers;
  tp_size_ = tp_size;
  current_counter_id_ = 0;

  // Initialize eventfds
  enable_eventfd_ = (layer_eventfds_tensor.numel() > 0);
  if (enable_eventfd_) {
    // layer_eventfds_tensor layout: [num_counters, tp_size, num_layers]
    // Index formula: counter_id * tp_size * num_layers + tp_rank * num_layers + layer
    int total_fds = layer_eventfds_tensor.numel();
    num_counters_ = total_fds / (tp_size * num_layers);
    
    int32_t *fds_ptr = layer_eventfds_tensor.data_ptr<int32_t>();
    layer_eventfds_.assign(fds_ptr, fds_ptr + total_fds);
    
    printf("[LayerwiseTransferGroup] Initialized with eventfds: "
           "tp_size=%d, num_counters=%d, num_layers=%d, total_fds=%d\n",
           tp_size_, num_counters_, num_layers_, total_fds);
  } else {
    num_counters_ = 0;
    printf("[LayerwiseTransferGroup] Initialized without eventfds\n");
  }

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];

  int64_t *kv_strides_ptr = gpu_kv_strides_tensor.data_ptr<int64_t>();
  int64_t *block_strides_ptr = gpu_block_strides_tensor.data_ptr<int64_t>();
  int64_t *layer_strides_ptr = gpu_layer_strides_tensor.data_ptr<int64_t>();
  int64_t *chunk_sizes_ptr = gpu_chunk_sizes_tensor.data_ptr<int64_t>();

  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = kv_strides_ptr[i];
    gpu_block_strides_in_bytes_[i] = block_strides_ptr[i];
    gpu_chunk_sizes_in_bytes_[i] = chunk_sizes_ptr[i];
    gpu_layer_strides_in_bytes_[i] = layer_strides_ptr[i];
  }

  num_tensors_per_gpu_ = gpu_blocks[0].size();
  cudaMallocHost((void **)&gpu_blocks_,
                 num_gpus_ * num_tensors_per_gpu_ * sizeof(void *));
  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_tensors_per_gpu_; ++j) {
      gpu_blocks_[i * num_tensors_per_gpu_ + j] = gpu_blocks[i][j].data_ptr();
    }
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

  cpu_blocks_ = cpu_blocks.data_ptr();

  // Get GPU device IDs from tensors (like tp_transfer_thread_group.cpp)
  gpu_device_ids_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gpu_device_ids_[i] = gpu_blocks[i][0].device().index();
  }

  // Create CUDA streams for each GPU
  streams_.resize(num_gpus_);
  events_.resize(num_gpus_);
  
  // Get highest priority (lowest value)
  int leastPriority, greatestPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  
  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(gpu_device_ids_[i]);
    cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, greatestPriority);
    cudaEventCreate(&events_[i]);
  }

  const char *issue_worker_env = std::getenv("FLEXKV_LAYERWISE_PERSISTENT_GPU_ISSUE");
  use_persistent_issue_workers_ = issue_worker_env != nullptr && std::string(issue_worker_env) == "1";

  // Initialize SSD IO context if ssd_files is not empty
  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }

  // Initialize indexer fuse support
  enable_indexer_ = !indexer_gpu_blocks.empty();
  if (enable_indexer_) {
    indexer_num_tensors_per_gpu_ = indexer_gpu_blocks[0].size();
    cudaMallocHost((void **)&indexer_gpu_blocks_,
                   num_gpus_ * indexer_num_tensors_per_gpu_ * sizeof(void *));
    for (int i = 0; i < num_gpus_; ++i) {
      for (int j = 0; j < indexer_num_tensors_per_gpu_; ++j) {
        indexer_gpu_blocks_[i * indexer_num_tensors_per_gpu_ + j] =
            indexer_gpu_blocks[i][j].data_ptr();
      }
    }

    indexer_cpu_blocks_ = indexer_cpu_blocks.data_ptr();

    indexer_gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
    indexer_gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
    indexer_gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
    indexer_gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];

    int64_t *idx_kv_strides_ptr = indexer_gpu_kv_strides_tensor.data_ptr<int64_t>();
    int64_t *idx_block_strides_ptr = indexer_gpu_block_strides_tensor.data_ptr<int64_t>();
    int64_t *idx_layer_strides_ptr = indexer_gpu_layer_strides_tensor.data_ptr<int64_t>();
    int64_t *idx_chunk_sizes_ptr = indexer_gpu_chunk_sizes_tensor.data_ptr<int64_t>();

    for (int i = 0; i < num_gpus; i++) {
      indexer_gpu_kv_strides_in_bytes_[i] = idx_kv_strides_ptr[i];
      indexer_gpu_block_strides_in_bytes_[i] = idx_block_strides_ptr[i];
      indexer_gpu_layer_strides_in_bytes_[i] = idx_layer_strides_ptr[i];
      indexer_gpu_chunk_sizes_in_bytes_[i] = idx_chunk_sizes_ptr[i];
    }

    // Determine indexer backend type from tensor count (symmetric with main KV)
    if (indexer_num_tensors_per_gpu_ == 1) {
      indexer_backend_type_ = BackendType::TRTLLM;
    } else if (indexer_num_tensors_per_gpu_ == num_layers) {
      indexer_backend_type_ = BackendType::VLLM;
    } else if (indexer_num_tensors_per_gpu_ == num_layers * 2) {
      indexer_backend_type_ = BackendType::SGLANG;
    } else {
      throw std::runtime_error("Unsupported indexer GPU block type: " +
                               std::to_string(indexer_num_tensors_per_gpu_));
    }

    // Build GTensorHandlers for indexer (symmetric with main KV)
    indexer_gpu_tensor_handlers_.reserve(num_gpus_);
    for (int i = 0; i < num_gpus_; i++) {
      int64_t **idx_gpu_blocks_ptr = reinterpret_cast<int64_t **>(
          indexer_gpu_blocks_ + i * indexer_num_tensors_per_gpu_);
      indexer_gpu_tensor_handlers_.emplace_back(
          indexer_backend_type_, idx_gpu_blocks_ptr, num_layers,
          indexer_gpu_kv_strides_in_bytes_[i],
          indexer_gpu_block_strides_in_bytes_[i],
          indexer_gpu_layer_strides_in_bytes_[i]);
    }

    fprintf(stderr, "[LayerwiseTransferGroup] Indexer fuse: enabled=true, "
           "num_tensors_per_gpu=%d, chunk_size=%ld bytes, backend=%s\n",
           indexer_num_tensors_per_gpu_, indexer_gpu_chunk_sizes_in_bytes_[0],
           indexer_backend_type_ == BackendType::SGLANG ? "SGLANG" :
           indexer_backend_type_ == BackendType::VLLM ? "VLLM" : "TRTLLM");
  }

  // Initialize indexer SSD IO context if indexer_ssd_files is not empty
  enable_indexer_ssd_ = !indexer_ssd_files.empty();
  if (enable_indexer_ssd_) {
    indexer_ioctx_ = std::make_unique<SSDIOCTX>(
        indexer_ssd_files, indexer_ssd_files.size(),
        iouring_entries, iouring_flags);
  }
}

LayerwiseTransferGroup::~LayerwiseTransferGroup() {
  // Stop polling thread first
  poll_stop_.store(true, std::memory_order_release);
  if (poll_thread_.joinable()) {
    poll_thread_.join();
  }
  // Clean up poll batch events
  for (auto &pb : poll_batches_) {
    for (auto &ev : pb.per_gpu_events) {
      cudaEventDestroy(ev);
    }
  }
  poll_batches_.clear();

  stop_issue_workers();
  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(gpu_device_ids_[i]);
    cudaStreamDestroy(streams_[i]);
    cudaEventDestroy(events_[i]);
  }

  cudaFreeHost(gpu_blocks_);

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;

  // Clean up indexer resources
  if (enable_indexer_) {
    cudaFreeHost(indexer_gpu_blocks_);
    indexer_gpu_tensor_handlers_.clear();
    delete[] indexer_gpu_kv_strides_in_bytes_;
    delete[] indexer_gpu_block_strides_in_bytes_;
    delete[] indexer_gpu_layer_strides_in_bytes_;
    delete[] indexer_gpu_chunk_sizes_in_bytes_;
  }
}

void LayerwiseTransferGroup::start_issue_workers_if_needed() {
  if (!use_persistent_issue_workers_ || issue_workers_started_) {
    return;
  }
  issue_workers_.clear();
  issue_workers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    auto state = std::make_unique<IssueWorkerState>();
    IssueWorkerState *raw = state.get();
    raw->thread = std::thread([this, raw, i]() {
      cudaSetDevice(gpu_device_ids_[i]);
      while (true) {
        std::function<void()> job;
        {
          std::unique_lock<std::mutex> lock(raw->mutex);
          raw->cv.wait(lock, [&]() { return raw->stop || raw->has_job; });
          if (raw->stop && !raw->has_job) {
            return;
          }
          job = std::move(raw->job);
          raw->has_job = false;
          raw->done = false;
          raw->exception = nullptr;
        }
        try {
          job();
        } catch (...) {
          raw->exception = std::current_exception();
        }
        {
          std::lock_guard<std::mutex> lock(raw->mutex);
          raw->done = true;
        }
        raw->done_cv.notify_one();
      }
    });
    issue_workers_.push_back(std::move(state));
  }
  issue_workers_started_ = true;
  fprintf(stderr, "[LW-WORKER] started persistent per-GPU issue workers: num_gpus=%d\n", num_gpus_);
}

void LayerwiseTransferGroup::stop_issue_workers() {
  if (!issue_workers_started_) {
    return;
  }
  for (auto &state : issue_workers_) {
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->stop = true;
    }
    state->cv.notify_one();
  }
  for (auto &state : issue_workers_) {
    if (state->thread.joinable()) {
      state->thread.join();
    }
  }
  issue_workers_.clear();
  issue_workers_started_ = false;
}

void LayerwiseTransferGroup::submit_issue_job(int gpu_idx, std::function<void()> job) {
  auto &state = issue_workers_[gpu_idx];
  std::unique_lock<std::mutex> lock(state->mutex);
  state->done_cv.wait(lock, [&]() { return state->done && !state->has_job; });
  state->job = std::move(job);
  state->done = false;
  state->has_job = true;
  state->exception = nullptr;
  lock.unlock();
  state->cv.notify_one();
}

void LayerwiseTransferGroup::wait_issue_job(int gpu_idx) {
  auto &state = issue_workers_[gpu_idx];
  std::unique_lock<std::mutex> lock(state->mutex);
  state->done_cv.wait(lock, [&]() { return state->done; });
  if (state->exception) {
    std::rethrow_exception(state->exception);
  }
}

void LayerwiseTransferGroup::layerwise_transfer(
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids_d2h,
    const int64_t ssd_layer_stride_in_bytes,
    const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
    const int round_robin, const int num_threads_per_device,
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_chunk_size_in_bytes,
    const int64_t h2d_cpu_kv_stride_in_bytes,
    const int64_t h2d_cpu_layer_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_cta_num,
    const bool use_ce_transfer, const int num_layers,
    const int layer_granularity, const bool is_mla,
    const int counter_id,
    const torch::Tensor &indexer_gpu_block_id_tensor,
    const torch::Tensor &indexer_cpu_block_id_tensor,
    const int64_t indexer_cpu_block_stride_in_bytes,
    const int64_t indexer_cpu_layer_stride_in_bytes,
    const int64_t indexer_h2d_cpu_kv_stride_in_bytes,
    const int64_t indexer_h2d_cpu_layer_stride_in_bytes,
    const torch::Tensor &indexer_ssd_block_ids,
    const torch::Tensor &indexer_cpu_block_ids_d2h,
    const int64_t indexer_ssd_layer_stride_in_bytes,
    const int64_t indexer_ssd_kv_stride_in_bytes,
    const int64_t indexer_cpu_chunk_size_in_bytes,
    const int indexer_num_blocks_per_file) {

  // Set current counter ID for eventfd notification
  current_counter_id_ = counter_id;
  start_issue_workers_if_needed();

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = cpu_blocks_;

  // Indexer block ids (may be empty if indexer is not enabled or not provided)
  bool do_indexer_transfer = enable_indexer_ &&
      indexer_gpu_block_id_tensor.defined() &&
      indexer_gpu_block_id_tensor.numel() > 0;
  int num_indexer_blocks = 0;
  int64_t *indexer_gpu_block_ids = nullptr;
  int64_t *indexer_cpu_block_ids = nullptr;
  if (do_indexer_transfer) {
    num_indexer_blocks = indexer_gpu_block_id_tensor.numel();
    indexer_gpu_block_ids =
        static_cast<int64_t *>(indexer_gpu_block_id_tensor.data_ptr());
    indexer_cpu_block_ids =
        static_cast<int64_t *>(indexer_cpu_block_id_tensor.data_ptr());
  }

  // Create CUDA events for timing each layer batch (on GPU 0)
  int num_batches = (num_layers + layer_granularity - 1) / layer_granularity;
  std::vector<int> batch_start_layers(num_batches);
  std::vector<int> batch_layers_count(num_batches);

  // Step 0: SSD -> CPU transfer for ALL layers at once (before layerwise loop).
  if (enable_ssd_ && ssd_block_ids.numel() > 0) {
    int num_ssd_blocks = ssd_block_ids.numel();
    int64_t ssd_bytes = cpu_chunk_size_in_bytes * 2 * num_layers * num_ssd_blocks;
    double ssd_mb = ssd_bytes / (1024.0 * 1024.0);
    char ssd_range_name[128];
    snprintf(ssd_range_name, sizeof(ssd_range_name),
             "SSD->CPU AllLayers[0,%d) %.2fMB", num_layers, ssd_mb);
    nvtxRangePushA(ssd_range_name);

    torch::Tensor all_layer_ids =
        torch::arange(0, num_layers,
                      torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *ioctx_, all_layer_ids, reinterpret_cast<int64_t>(cpu_blocks_),
        ssd_block_ids, cpu_block_ids_d2h, cpu_layer_stride_in_bytes,
        cpu_kv_stride_in_bytes, ssd_layer_stride_in_bytes,
        ssd_kv_stride_in_bytes, cpu_chunk_size_in_bytes,
        cpu_block_stride_in_bytes,
        true, // is_read: SSD -> CPU
        num_blocks_per_file, round_robin, num_threads_per_device, is_mla);

    nvtxRangePop();
  }

  // Indexer SSD -> CPU transfer for ALL layers at once.
  if (enable_indexer_ssd_ && indexer_ssd_block_ids.defined() &&
      indexer_ssd_block_ids.numel() > 0) {
    int num_indexer_ssd_blocks = indexer_ssd_block_ids.numel();
    int64_t indexer_ssd_bytes = indexer_cpu_chunk_size_in_bytes * num_layers * num_indexer_ssd_blocks;
    double indexer_ssd_mb = indexer_ssd_bytes / (1024.0 * 1024.0);
    char idx_ssd_range_name[128];
    snprintf(idx_ssd_range_name, sizeof(idx_ssd_range_name),
             "Indexer SSD->CPU AllLayers[0,%d) %.2fMB", num_layers, indexer_ssd_mb);
    nvtxRangePushA(idx_ssd_range_name);

    torch::Tensor all_layer_ids =
        torch::arange(0, num_layers,
                      torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *indexer_ioctx_, all_layer_ids,
        reinterpret_cast<int64_t>(indexer_cpu_blocks_),
        indexer_ssd_block_ids, indexer_cpu_block_ids_d2h,
        indexer_cpu_layer_stride_in_bytes,
        indexer_ssd_kv_stride_in_bytes,
        indexer_ssd_layer_stride_in_bytes,
        indexer_ssd_kv_stride_in_bytes,
        indexer_cpu_chunk_size_in_bytes,
        indexer_cpu_block_stride_in_bytes,
        true, // is_read: SSD -> CPU
        indexer_num_blocks_per_file, round_robin, num_threads_per_device,
        true /* is_mla: indexer always MLA */);

    nvtxRangePop();
  }

  // ===== Event-based layerwise notification =====
  // Prepare per-batch polling structures. For each batch, we create one
  // cudaEvent per GPU. After submitting all GPU transfers for a batch, we
  // record an event on each GPU stream. The polling thread queries these
  // events and writes eventfds as soon as a batch completes on all GPUs.

  // If a previous poll thread is still running, join it first to ensure
  // all GPU work from the previous transfer is complete.
  if (poll_active_) {
    poll_stop_.store(true, std::memory_order_release);
    poll_cv_.notify_all();
    if (poll_thread_.joinable()) {
      poll_thread_.join();
    }
    poll_active_ = false;
  }

  // Clean up previous poll batches' events
  for (auto &pb : poll_batches_) {
    for (auto &ev : pb.per_gpu_events) {
      cudaEventDestroy(ev);
    }
  }
  poll_batches_.clear();
  poll_batches_.resize(num_batches);
  for (int b = 0; b < num_batches; ++b) {
    int sl = b * layer_granularity;
    int ltb = std::min(layer_granularity, num_layers - sl);
    poll_batches_[b].start_layer = sl;
    poll_batches_[b].layers_this_batch = ltb;
    poll_batches_[b].per_gpu_events.resize(num_gpus_);
    poll_batches_[b].notified = false;
    for (int g = 0; g < num_gpus_; ++g) {
      cudaSetDevice(gpu_device_ids_[g]);
      cudaEventCreateWithFlags(&poll_batches_[b].per_gpu_events[g],
                                cudaEventDisableTiming);
    }
  }
  poll_stop_.store(false, std::memory_order_release);
  poll_next_batch_.store(0, std::memory_order_release);

  int batch_idx = 0;
  for (int start_layer = 0; start_layer < num_layers;
       start_layer += layer_granularity) {
    int layers_this_batch =
        std::min(layer_granularity, num_layers - start_layer);

    batch_start_layers[batch_idx] = start_layer;
    batch_layers_count[batch_idx] = layers_this_batch;

    // Step 1: CPU -> GPU transfer for this layer batch
    auto issue_one_gpu = [&](int i) {
      cudaSetDevice(gpu_device_ids_[i]);
      int64_t cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
      if (is_mla) {
        cpu_startoff_inside_chunks = 0;
      }
      int64_t gpu_startoff_inside_chunks = 0;
      int64_t chunk_size = gpu_chunk_sizes_in_bytes_[i];

      switch (backend_type_) {
      case BackendType::VLLM:
        flexkv::transfer_kv_blocks<BackendType::VLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla, false);
        break;
      case BackendType::TRTLLM:
        flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla, false);
        break;
      case BackendType::SGLANG:
        flexkv::transfer_kv_blocks<BackendType::SGLANG>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla, false);
        break;
      }

      // Fused indexer CPU -> GPU transfer on the same stream
      if (do_indexer_transfer) {
        int64_t idx_chunk_size = indexer_gpu_chunk_sizes_in_bytes_[i];
        int64_t idx_cpu_startoff = 0;

        switch (indexer_backend_type_) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_indexer_blocks, start_layer, layers_this_batch,
              indexer_gpu_block_ids, indexer_gpu_tensor_handlers_[i],
              0, indexer_cpu_block_ids, indexer_cpu_blocks_,
              indexer_h2d_cpu_kv_stride_in_bytes,
              indexer_h2d_cpu_layer_stride_in_bytes,
              indexer_cpu_block_stride_in_bytes,
              idx_cpu_startoff, idx_chunk_size,
              streams_[i], transfer_cta_num, true, use_ce_transfer, true, false);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_indexer_blocks, start_layer, layers_this_batch,
              indexer_gpu_block_ids, indexer_gpu_tensor_handlers_[i],
              0, indexer_cpu_block_ids, indexer_cpu_blocks_,
              indexer_h2d_cpu_kv_stride_in_bytes,
              indexer_h2d_cpu_layer_stride_in_bytes,
              indexer_cpu_block_stride_in_bytes,
              idx_cpu_startoff, idx_chunk_size,
              streams_[i], transfer_cta_num, true, use_ce_transfer, true, false);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_indexer_blocks, start_layer, layers_this_batch,
              indexer_gpu_block_ids, indexer_gpu_tensor_handlers_[i],
              0, indexer_cpu_block_ids, indexer_cpu_blocks_,
              indexer_h2d_cpu_kv_stride_in_bytes,
              indexer_h2d_cpu_layer_stride_in_bytes,
              indexer_cpu_block_stride_in_bytes,
              idx_cpu_startoff, idx_chunk_size,
              streams_[i], transfer_cta_num, true, use_ce_transfer, true, false);
          break;
        }
      }
    };

    if (use_persistent_issue_workers_) {
      for (int i = 0; i < num_gpus_; ++i) {
        submit_issue_job(i, [&, i]() { issue_one_gpu(i); });
      }
      for (int i = 0; i < num_gpus_; ++i) {
        wait_issue_job(i);
      }
    } else {
      for (int i = 0; i < num_gpus_; ++i) {
        issue_one_gpu(i);
      }
    }

    // Record a CUDA event on each GPU stream after this batch's transfers.
    // The polling thread will query these events to detect batch completion
    // and write eventfds for true layerwise notification.
    for (int g = 0; g < num_gpus_; ++g) {
      cudaSetDevice(gpu_device_ids_[g]);
      cudaEventRecord(poll_batches_[batch_idx].per_gpu_events[g], streams_[g]);
    }

    batch_idx++;
  }

  // Start the polling thread. It will query events and write eventfds
  // as each batch completes, enabling overlap with scheduler compute.
  // The main thread returns immediately (no cudaStreamSynchronize).
  poll_active_ = true;
  poll_thread_ = std::thread(&LayerwiseTransferGroup::event_polling_loop, this);

  fprintf(stderr, "[LW-EVT] submitted %d batches, polling thread launched, "
          "main thread returning (no sync)\n", num_batches);
}

} // namespace flexkv
