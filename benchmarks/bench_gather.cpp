/*
 * Standalone microbenchmark for CPU gather (scattered CPU blocks -> contiguous
 * staging), the bottleneck of long-context + CE-mode KV loads in FlexKV.
 *
 * It compares four strategies on the SAME access pattern:
 *   [1] single-thread memcpy        (what FlexKV Path 2 cpu_gather did before)
 *   [2] parallel memcpy             (P1 threading only)
 *   [3] parallel + NT-store         (P1 full: AVX-512/AVX-2 streaming stores)
 *   [4] Path 0 big contiguous copy  (what P2 contiguous allocation unlocks)
 *
 * The NT-store + dispatch logic mirrors csrc/parallel_copy.cpp so the numbers
 * are representative of the real implementation.
 *
 * Build (no CUDA needed):
 *   g++ -O3 -std=c++17 -pthread bench_gather.cpp -o bench_gather
 *   # (AVX paths use function target-attributes + runtime dispatch, so
 *   #  -march=native is NOT required; add it only if you want the scalar
 *   #  fallback / head-tail code to vectorize too.)
 *
 * Run (defaults model an MLA 100K-token load):
 *   ./bench_gather
 *   # or override: num_blocks chunk_bytes reps nthreads scatter(0|1|2)
 *   ./bench_gather 1560 73728 60 16 0
 *     scatter: 0=fully scattered  1=segmented(runs)  2=contiguous
 *
 * Env knobs:
 *   BENCH_POOL_EXPAND=N   pool = num_blocks*N blocks (default 4, to exceed LLC)
 *   BENCH_NUM_SEGMENTS=N  number of contiguous runs for scatter mode 1 (def 32)
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define BENCH_X86 1
#else
#define BENCH_X86 0
#endif

// ---------------------------------------------------------------------------
// NT-store streaming copy (alignment-safe), mirroring parallel_copy.cpp.
// ---------------------------------------------------------------------------
enum class Impl { SCALAR, AVX2, AVX512 };

#if BENCH_X86
__attribute__((target("avx512f"))) static void nt_copy_avx512(char *d,
                                                              const char *s,
                                                              size_t n) {
  size_t i = 0;
  size_t head = (size_t)((64 - ((uintptr_t)d & 63)) & 63);
  if (head > n) head = n;
  if (head) { std::memcpy(d, s, head); i = head; }
  for (; i + 64 <= n; i += 64)
    _mm512_stream_si512((__m512i *)(d + i),
                        _mm512_loadu_si512((const void *)(s + i)));
  if (i < n) std::memcpy(d + i, s + i, n - i);
}
__attribute__((target("avx2"))) static void nt_copy_avx2(char *d, const char *s,
                                                         size_t n) {
  size_t i = 0;
  size_t head = (size_t)((32 - ((uintptr_t)d & 31)) & 31);
  if (head > n) head = n;
  if (head) { std::memcpy(d, s, head); i = head; }
  for (; i + 32 <= n; i += 32)
    _mm256_stream_si256((__m256i *)(d + i),
                        _mm256_loadu_si256((const __m256i *)(s + i)));
  if (i < n) std::memcpy(d + i, s + i, n - i);
}
#endif

static Impl detect_impl() {
#if BENCH_X86
  if (__builtin_cpu_supports("avx512f")) return Impl::AVX512;
  if (__builtin_cpu_supports("avx2")) return Impl::AVX2;
#endif
  return Impl::SCALAR;
}
static const char *impl_name(Impl i) {
  switch (i) {
  case Impl::AVX512: return "AVX-512 NT";
  case Impl::AVX2: return "AVX-2 NT";
  default: return "scalar memcpy";
  }
}
static inline void nt_copy(char *d, const char *s, size_t n, Impl impl) {
#if BENCH_X86
  if (impl == Impl::AVX512) { nt_copy_avx512(d, s, n); return; }
  if (impl == Impl::AVX2) { nt_copy_avx2(d, s, n); return; }
#endif
  std::memcpy(d, s, n);
}
static inline void nt_fence() {
#if BENCH_X86
  _mm_sfence();
#endif
}

// ---------------------------------------------------------------------------
// Simple fork-join parallel_for. Spawn overhead (~10-20us) is negligible vs the
// 100MB+ per-rep copies measured here.
// ---------------------------------------------------------------------------
static void parallel_for(int total, int nthreads,
                         const std::function<void(int, int)> &body) {
  if (nthreads <= 1 || total < nthreads) { body(0, total); return; }
  int chunk = (total + nthreads - 1) / nthreads;
  std::vector<std::thread> ths;
  ths.reserve(nthreads - 1);
  for (int t = 1; t < nthreads; ++t) {
    int s = t * chunk, e = std::min(s + chunk, total);
    if (s >= total) break;
    ths.emplace_back([&body, s, e] { body(s, e); });
  }
  body(0, std::min(chunk, total));
  for (auto &th : ths) th.join();
}

// ---------------------------------------------------------------------------
// Timing harness: returns median ms over `reps`.
// ---------------------------------------------------------------------------
template <typename F> static double time_ms_median(int reps, F &&fn) {
  std::vector<double> samples;
  samples.reserve(reps);
  fn(); // warmup
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int main(int argc, char **argv) {
  // ---- config ----
  int num_blocks = (argc > 1) ? std::atoi(argv[1]) : 1560;    // ~100K tok / page64
  size_t chunk = (argc > 2) ? (size_t)std::atoll(argv[2]) : 73728; // MLA chunk
  int reps = (argc > 3) ? std::atoi(argv[3]) : 60;            // ~num_layers
  int nthreads = (argc > 4) ? std::atoi(argv[4]) : 0;
  int scatter = (argc > 5) ? std::atoi(argv[5]) : 0;          // 0 scat,1 seg,2 contig
  if (nthreads <= 0) {
    unsigned hw = std::thread::hardware_concurrency();
    nthreads = hw ? (int)std::min<unsigned>(hw, 32u) : 8;
    nthreads = std::max(1, nthreads / 2);
  }
  int pool_expand = std::getenv("BENCH_POOL_EXPAND")
                        ? std::atoi(std::getenv("BENCH_POOL_EXPAND")) : 4;
  if (pool_expand < 1) pool_expand = 1;
  int num_segments = std::getenv("BENCH_NUM_SEGMENTS")
                         ? std::atoi(std::getenv("BENCH_NUM_SEGMENTS")) : 32;

  const Impl impl = detect_impl();
  size_t num_total_blocks = (size_t)num_blocks * pool_expand;
  size_t pool_bytes = num_total_blocks * chunk;
  size_t staging_bytes = (size_t)num_blocks * chunk;
  double moved_per_rep_gb = (double)num_blocks * chunk / 1e9;

  printf("==== CPU gather microbenchmark ====\n");
  printf("num_blocks=%d  chunk=%zuB  reps=%d  threads=%d  scatter=%s\n",
         num_blocks, chunk, reps, nthreads,
         scatter == 0 ? "scattered" : scatter == 1 ? "segmented" : "contiguous");
  printf("pool=%.2f GB (%zu blocks)  staging=%.2f MB  SIMD=%s  chunk%%64=%zu\n",
         pool_bytes / 1e9, num_total_blocks, staging_bytes / 1e6,
         impl_name(impl), chunk % 64);
  printf("moved/rep=%.2f MB  (gather copies num_blocks*chunk per rep)\n\n",
         moved_per_rep_gb * 1e3);

  // ---- allocate (64B aligned) ----
  auto aligned = [](size_t n) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, n) != 0) { perror("posix_memalign"); exit(1); }
    return (char *)p;
  };
  char *pool = aligned(pool_bytes);
  char *staging = aligned(staging_bytes);
  char *ref = aligned(staging_bytes);
  std::memset(pool, 1, pool_bytes); // fault-in
  std::memset(staging, 0, staging_bytes);

  // ---- block id pattern ----
  std::vector<int64_t> ids(num_blocks);
  if (scatter == 2) {
    for (int k = 0; k < num_blocks; ++k) ids[k] = k; // fully contiguous
  } else if (scatter == 1) {
    // num_segments contiguous runs placed at random non-overlapping offsets
    int seg = std::max(1, num_segments);
    int per = (num_blocks + seg - 1) / seg;
    std::mt19937_64 rng(123);
    int written = 0;
    for (int s = 0; s < seg && written < num_blocks; ++s) {
      int run = std::min(per, num_blocks - written);
      int64_t base = (int64_t)(rng() % (num_total_blocks - run));
      for (int j = 0; j < run; ++j) ids[written++] = base + j;
    }
  } else {
    // fully scattered: random distinct ids across the (larger) pool
    std::vector<int64_t> all(num_total_blocks);
    for (size_t i = 0; i < num_total_blocks; ++i) all[i] = (int64_t)i;
    std::mt19937_64 rng(123);
    std::shuffle(all.begin(), all.end(), rng);
    for (int k = 0; k < num_blocks; ++k) ids[k] = all[k];
  }
  // detect whether ids are fully contiguous (Path 0 eligible)
  bool contig = true;
  for (int k = 1; k < num_blocks; ++k)
    if (ids[k] != ids[k - 1] + 1) { contig = false; break; }

  const int64_t *bid = ids.data();

  // ---- reference (correctness) ----
  for (int k = 0; k < num_blocks; ++k)
    std::memcpy(ref + (size_t)k * chunk, pool + (size_t)bid[k] * chunk, chunk);
  auto verify = [&](const char *name) {
    if (std::memcmp(ref, staging, staging_bytes) != 0)
      printf("  [WARN] %s produced INCORRECT output!\n", name);
    std::memset(staging, 0, staging_bytes);
  };

  struct Row { const char *name; double ms; };
  std::vector<Row> rows;

  // [1] single-thread memcpy
  rows.push_back({"[1] single-thread memcpy",
      time_ms_median(reps, [&] {
        for (int k = 0; k < num_blocks; ++k)
          std::memcpy(staging + (size_t)k * chunk,
                      pool + (size_t)bid[k] * chunk, chunk);
      })});
  verify(rows.back().name);

  // [2] parallel memcpy (no NT)
  rows.push_back({"[2] parallel memcpy",
      time_ms_median(reps, [&] {
        parallel_for(num_blocks, nthreads, [&](int b, int e) {
          for (int k = b; k < e; ++k)
            std::memcpy(staging + (size_t)k * chunk,
                        pool + (size_t)bid[k] * chunk, chunk);
        });
      })});
  verify(rows.back().name);

  // [3] parallel + NT-store
  rows.push_back({"[3] parallel + NT-store",
      time_ms_median(reps, [&] {
        parallel_for(num_blocks, nthreads, [&](int b, int e) {
          for (int k = b; k < e; ++k) {
            if (k + 16 < e)
              __builtin_prefetch(pool + (size_t)bid[k + 16] * chunk, 0, 0);
            nt_copy(staging + (size_t)k * chunk,
                    pool + (size_t)bid[k] * chunk, chunk, impl);
          }
          nt_fence();
        });
      })});
  verify(rows.back().name);

  // [4] Path 0: one big contiguous copy (only valid if ids contiguous)
  if (contig) {
    rows.push_back({"[4] Path 0 big memcpy (1 thread)",
        time_ms_median(reps, [&] {
          std::memcpy(staging, pool + (size_t)bid[0] * chunk,
                      (size_t)num_blocks * chunk);
        })});
    verify(rows.back().name);
    rows.push_back({"[4b] Path 0 big copy (parallel NT)",
        time_ms_median(reps, [&] {
          const char *src = pool + (size_t)bid[0] * chunk;
          parallel_for(num_blocks, nthreads, [&](int b, int e) {
            size_t off = (size_t)b * chunk;
            nt_copy(staging + off, src + off, (size_t)(e - b) * chunk, impl);
            nt_fence();
          });
        })});
    verify(rows.back().name);
  }

  // ---- report ----
  printf("%-36s %10s %12s %10s\n", "method", "ms/rep", "GB/s", "speedup");
  double base = rows[0].ms;
  for (auto &r : rows) {
    double gbps = moved_per_rep_gb / (r.ms / 1e3);
    printf("%-36s %10.3f %12.2f %9.2fx\n", r.name, r.ms, gbps, base / r.ms);
  }
  if (!contig)
    printf("\n(Path 0 skipped: ids not contiguous. Run with scatter=2, or rely\n"
           " on P2 contiguous allocation to make real loads contiguous.)\n");

  free(pool); free(staging); free(ref);
  return 0;
}
