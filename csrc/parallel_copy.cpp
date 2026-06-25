/*
 * [P1] Parallel CPU gather/scatter implementation.
 * Compiled by the host C++ compiler (not nvcc), so AVX intrinsics are fine.
 * AVX512/AVX2 paths use GCC/Clang target attributes + runtime CPU dispatch,
 * so no global -mavx* build flags are required and the code stays portable
 * (non-x86 falls back to std::memcpy).
 */
#include "parallel_copy.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FLEXKV_X86 1
#else
#define FLEXKV_X86 0
#endif

namespace flexkv {
namespace {

int get_env_int(const char *name, int def) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return def;
  return std::atoi(e);
}

bool parallel_enabled() {
  static const int v = get_env_int("FLEXKV_PARALLEL_GATHER", 1);
  return v != 0;
}

bool nt_enabled() {
  static const int v = get_env_int("FLEXKV_GATHER_NT", 1);
  return v != 0;
}

// Total participating threads (including the calling thread).
int configured_threads() {
  static const int v = []() {
    int n = get_env_int("FLEXKV_GATHER_THREADS", 0);
    if (n <= 0) {
      unsigned hw = std::thread::hardware_concurrency();
      int base = hw ? static_cast<int>(std::min<unsigned>(hw, 16u)) : 8;
      n = std::max(1, base / 2);
    }
    return std::max(1, n);
  }();
  return v;
}

// ---------------------------------------------------------------------------
// Non-temporal streaming copy (write side bypasses cache). Alignment-safe:
// the destination head is copied with memcpy until 64/32B aligned, the aligned
// middle uses NT stores, and the tail falls back to memcpy. Callers must issue
// one sfence after a batch of copies (see copy_range_*).
// ---------------------------------------------------------------------------
enum class CopyImpl { SCALAR, AVX2, AVX512 };

#if FLEXKV_X86
__attribute__((target("avx512f"))) void nt_copy_avx512(char *d, const char *s,
                                                       size_t n) {
  size_t i = 0;
  size_t head = static_cast<size_t>((64 - (reinterpret_cast<uintptr_t>(d) & 63)) & 63);
  if (head > n)
    head = n;
  if (head) {
    std::memcpy(d, s, head);
    i = head;
  }
  for (; i + 64 <= n; i += 64) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const void *>(s + i));
    _mm512_stream_si512(reinterpret_cast<__m512i *>(d + i), v);
  }
  if (i < n)
    std::memcpy(d + i, s + i, n - i);
}

__attribute__((target("avx2"))) void nt_copy_avx2(char *d, const char *s,
                                                  size_t n) {
  size_t i = 0;
  size_t head = static_cast<size_t>((32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31);
  if (head > n)
    head = n;
  if (head) {
    std::memcpy(d, s, head);
    i = head;
  }
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i));
    _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i), v);
  }
  if (i < n)
    std::memcpy(d + i, s + i, n - i);
}
#endif

CopyImpl pick_impl() {
#if FLEXKV_X86
  static const CopyImpl impl = []() {
    if (!nt_enabled())
      return CopyImpl::SCALAR;
    if (__builtin_cpu_supports("avx512f"))
      return CopyImpl::AVX512;
    if (__builtin_cpu_supports("avx2"))
      return CopyImpl::AVX2;
    return CopyImpl::SCALAR;
  }();
  return impl;
#else
  return CopyImpl::SCALAR;
#endif
}

inline void copy_one(char *d, const char *s, size_t n, CopyImpl impl) {
#if FLEXKV_X86
  switch (impl) {
  case CopyImpl::AVX512:
    nt_copy_avx512(d, s, n);
    return;
  case CopyImpl::AVX2:
    nt_copy_avx2(d, s, n);
    return;
  default:
    std::memcpy(d, s, n);
    return;
  }
#else
  (void)impl;
  std::memcpy(d, s, n);
#endif
}

inline void fence_if_nt(CopyImpl impl) {
#if FLEXKV_X86
  if (impl != CopyImpl::SCALAR)
    _mm_sfence();
#else
  (void)impl;
#endif
}

// ---------------------------------------------------------------------------
// Persistent per-process CPU thread pool. The calling thread participates as
// one worker; (configured_threads-1) background threads handle the rest.
// cv-based wakeup (not busy-spin) — gather is ms-scale for long context, so
// wakeup latency is negligible and we avoid burning idle cores.
// ---------------------------------------------------------------------------
class CopyPool {
public:
  static CopyPool &instance() {
    static CopyPool p(configured_threads());
    return p;
  }

  int worker_count() const { return static_cast<int>(workers_.size()); }

  // Split [0,total) across background workers + the caller thread.
  void run(int total, const std::function<void(int, int)> &body) {
    const int nw = static_cast<int>(workers_.size());
    const int participants = nw + 1;
    const int chunk = (total + participants - 1) / participants;

    for (int w = 0; w < nw; ++w) {
      const int start = (w + 1) * chunk;
      const int end = std::min(start + chunk, total);
      if (start >= total) {
        workers_[w]->mark_done();
        continue;
      }
      workers_[w]->submit(&body, start, end);
    }
    const int main_end = std::min(chunk, total);
    if (main_end > 0)
      body(0, main_end);
    for (int w = 0; w < nw; ++w)
      workers_[w]->wait();
  }

private:
  struct Worker {
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    std::condition_variable done_cv;
    const std::function<void(int, int)> *fn = nullptr;
    int start = 0, end = 0;
    bool has_job = false;
    bool done = true;
    bool stop = false;

    Worker() { th = std::thread([this] { loop(); }); }
    ~Worker() {
      {
        std::lock_guard<std::mutex> lk(m);
        stop = true;
      }
      cv.notify_one();
      if (th.joinable())
        th.join();
    }
    void submit(const std::function<void(int, int)> *f, int s, int e) {
      {
        std::lock_guard<std::mutex> lk(m);
        fn = f;
        start = s;
        end = e;
        has_job = true;
        done = false;
      }
      cv.notify_one();
    }
    void mark_done() {
      std::lock_guard<std::mutex> lk(m);
      done = true;
    }
    void wait() {
      std::unique_lock<std::mutex> lk(m);
      done_cv.wait(lk, [this] { return done; });
    }
    void loop() {
      for (;;) {
        const std::function<void(int, int)> *f = nullptr;
        int s = 0, e = 0;
        {
          std::unique_lock<std::mutex> lk(m);
          cv.wait(lk, [this] { return has_job || stop; });
          if (stop && !has_job)
            return;
          f = fn;
          s = start;
          e = end;
          has_job = false;
        }
        try {
          (*f)(s, e);
        } catch (...) {
        }
        {
          std::lock_guard<std::mutex> lk(m);
          done = true;
        }
        done_cv.notify_one();
      }
    }
  };

  std::vector<std::unique_ptr<Worker>> workers_;
  explicit CopyPool(int total_threads) {
    for (int i = 0; i < total_threads - 1; ++i)
      workers_.push_back(std::make_unique<Worker>());
  }
};

constexpr int kPrefetchDist = 16;

} // namespace

void parallel_gather_blocks(void *staging, const void *src_base,
                            const int64_t *block_ids,
                            int64_t block_stride_bytes, int64_t chunk_bytes,
                            int num_blocks) {
  if (num_blocks <= 0)
    return;
  const CopyImpl impl = pick_impl();
  char *dst = static_cast<char *>(staging);
  const char *src = static_cast<const char *>(src_base);

  auto body = [&](int begin, int end) {
    for (int k = begin; k < end; ++k) {
      if (k + kPrefetchDist < end)
        __builtin_prefetch(src + block_ids[k + kPrefetchDist] * block_stride_bytes,
                           0 /*read*/, 0 /*non-temporal*/);
      copy_one(dst + static_cast<size_t>(k) * chunk_bytes,
               src + block_ids[k] * block_stride_bytes,
               static_cast<size_t>(chunk_bytes), impl);
    }
    fence_if_nt(impl);
  };

  if (!parallel_enabled() || num_blocks < 8 ||
      CopyPool::instance().worker_count() == 0) {
    body(0, num_blocks);
    return;
  }
  CopyPool::instance().run(num_blocks, body);
}

void parallel_scatter_blocks(void *dst_base, const int64_t *block_ids,
                             int64_t block_stride_bytes, const void *staging,
                             int64_t chunk_bytes, int num_blocks) {
  if (num_blocks <= 0)
    return;
  const CopyImpl impl = pick_impl();
  char *dst = static_cast<char *>(dst_base);
  const char *src = static_cast<const char *>(staging);

  auto body = [&](int begin, int end) {
    for (int k = begin; k < end; ++k) {
      if (k + kPrefetchDist < end)
        __builtin_prefetch(dst + block_ids[k + kPrefetchDist] * block_stride_bytes,
                           1 /*write*/, 0 /*non-temporal*/);
      copy_one(dst + block_ids[k] * block_stride_bytes,
               src + static_cast<size_t>(k) * chunk_bytes,
               static_cast<size_t>(chunk_bytes), impl);
    }
    fence_if_nt(impl);
  };

  if (!parallel_enabled() || num_blocks < 8 ||
      CopyPool::instance().worker_count() == 0) {
    body(0, num_blocks);
    return;
  }
  CopyPool::instance().run(num_blocks, body);
}

} // namespace flexkv
