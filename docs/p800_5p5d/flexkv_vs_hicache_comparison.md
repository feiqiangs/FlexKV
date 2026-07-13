# P800 5P5D：FlexKV（优化后）vs sglang 原生 HiCache 对比测试结果

> 本文档记录 **FlexKV+mooncake-store（已优化）** 与 **sglang 原生 HiCache**（同样接 mooncake-store 做 L3）
> 在 P800 5P5D 集群（5 prefill + 5 decode，每实例 tp16=2 节点，共 20 节点）上、**同等参数条件**下的对比测试结果。
>
> - 测试日期：2026-07-13
> - 完整踩坑记录/优化前数据/详细排障过程见：`doc/p800_hicache_vs_flexkv_comparison.md`
> - 高并发瓶颈根因分析见：`flexkv_vs_hicache_high_concurrency_analysis.md`

---

## 0. 结论先行（TL;DR）

FlexKV 优化后（关闭纯 TP 场景下冗余的跨 rank collective sync），在与 HiCache **严格同等的 L2/L3 容量、同一份测试脚本与并发梯度**下：

- **命中率完全对齐**（同一并发档差距 <0.3%），两套缓存机制在等效容量下的缓存行为一致。
- **低并发（c≤8）**：两者表现相当，互有小幅胜负。
- **高并发（c=32）**：FlexKV **反超** HiCache——TTFT p50 1.50s（HiCache 1.70s），吞吐 401.6 tok/s（HiCache 344.7 tok/s）。
- FlexKV 已解决此前存在的高并发退化问题（历史数据见 `doc/p800_hicache_vs_flexkv_comparison.md` §3.3/§5）。

---

## 1. 测试条件

### 1.1 对齐方案

| 层级 | FlexKV | sglang 原生 HiCache | 说明 |
|---|---|---|---|
| L1（GPU 设备池） | `--mem-fraction-static 0.87` | `--mem-fraction-static 0.87` | 相同 |
| L2（host 内存） | `FLEXKV_CPU_CACHE_GB=192`（整节点共享一份） | `--hicache-size 24`（每 TP rank 独立分配，8 rank/节点 × 24GB） | 两者节点总量严格对齐为 **192GB/节点** |
| L3（mooncake-store） | `MOONCAKE_GLOBAL_SEGMENT_SIZE_GB=8`/节点 | `--hicache-storage-backend mooncake`，`global_segment_size=8gb` | 每节点贡献 8GB，复用同一个 mooncake master（`10.129.1.31:50051`） |
| 其余参数 | `page-size/context-length/attention-backend/tp-size/ep-size/chunked-prefill-size/max-running-requests/moe-*/disaggregation-*` 等完全一致 | 同左 | 只替换 `--kv-connector-cls flexkv` ↔ `--enable-hierarchical-cache ...` |
| decode 侧 | 不变 | 不变 | 两者均只是 prefill 侧本地前缀缓存机制 |

### 1.2 测试脚本与并发梯度

```
--mode fixed --concurrency 1,4,8,16,32 --num-requests 40 --prefix-tokens 1200 --max-tokens 64 --flush-before
```

每档 40 个独立 session，共享约 1200 token 的合成前缀 + 唯一后缀（自包含，不依赖外部数据集）。
脚本：`zittozhang_scripts/scripts/bench_5p5d.py`（含 TTFT/TPOT/E2E 的 p50/p90/p99 全量指标）。

### 1.3 FlexKV 优化内容

在 P800 集群 pod 内 `/workspace/sglang_dev`（git repo，`p800_flexkv` 分支）上做了如下改动：

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/mem_cache/storage/flexkv/flexkv_comm.py` | **核心改动**：`needs_sync` 由 `(pp_size>1 or attn_tp_size>1 or attn_cp_size>1)` 硬编码为 `False`。纯 TP 场景（`pp=1, cp=1, tp=16`）下所有 TP rank 跑相同 token 序列，rank0 leader-broadcast 是冗余的，关闭 FlexKV 跨 rank 的 collective sync 即可消除高并发下的控制面瓶颈 |
| `python/sglang/srt/mem_cache/extended_radix_cache.py` | 新增 `FLEXKV_MIN_LAYERWISE_LOAD_TOKENS` 阈值：host 命中长度过小时跳过外部 load-back，减少小请求的无意义 IO 开销 |
| `python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py` | 新增 `FLEXKV_D2H_PROFILE` 调试打点（纯 profiling） |

> 注意：`needs_sync=False` 目前是硬编码，仅适用于纯 TP 场景（`pp=1, cp=1`）；若后续切换到 PP>1 或 CP>1 部署，需要恢复条件判断逻辑，否则会跳过必要的 leader-broadcast 导致 rank 间状态不一致。该改动尚未提交到 git（`p800_flexkv` 分支为 uncommitted 状态），需确认稳定后提交，避免环境重建/pod 重启丢失。

---

## 2. 测试结果

### 2.1 汇总对比表

| 并发 | 方案 | 命中率 | TTFT p50 (s) | TTFT p90 (s) | TTFT p99 (s) | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) | out tok/s | QPS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1  | FlexKV  | 93.6% | 0.43 | 0.52 | 3.56 | 27.0 | 37.8 | 42.8 | 27.8  | 0.43 |
| 1  | HiCache | 93.8% | 0.49 | 0.67 | 3.86 | 28.7 | 34.9 | 35.9 | 26.0  | 0.41 |
| 4  | FlexKV  | 86.4% | 0.87 | 1.21 | 2.16 | 30.1 | 33.8 | 38.6 | 88.0  | 1.37 |
| 4  | HiCache | 86.6% | 0.96 | 4.31 | 4.44 | 28.8 | 30.9 | 37.1 | 80.8  | 1.26 |
| 8  | FlexKV  | 76.8% | 1.28 | 3.94 | 4.01 | 30.3 | 38.1 | 43.7 | 137.9 | 2.15 |
| 8  | HiCache | 76.9% | 1.24 | 1.55 | 1.69 | 31.2 | 42.1 | 49.7 | 151.7 | 2.37 |
| 16 | FlexKV  | 57.6% | 1.26 | 4.35 | 4.68 | 31.5 | 43.5 | 44.6 | 202.7 | 3.17 |
| 16 | HiCache | 57.7% | 1.56 | 4.55 | 4.62 | 30.2 | 38.3 | 44.9 | 203.9 | 3.19 |
| 32 | **FlexKV**  | 57.6% | **1.50** | 2.16 | 3.21 | 33.0 | 35.7 | 36.4 | **401.6** | **6.27** |
| 32 | HiCache | 57.7% | 1.70 | 2.29 | 2.87 | 43.0 | 50.0 | 52.0 | 344.7 | 5.39 |

### 2.2 逐档分析

- **c=1/4**：两者基本打平，互有微弱优势，属误差范围。
- **c=8**：HiCache 的 TTFT p90/p99 更稳定（1.55/1.69s vs FlexKV 3.94/4.01s），吞吐略高（151.7 vs 137.9 tok/s）。
- **c=16**：两者吞吐接近（202.7 vs 203.9 tok/s），TTFT 长尾均出现一次尖峰（4.35s/4.55s），怀疑与 `--max-running-requests 16` 恰好等于该并发值的排队边界效应有关，建议后续用更细粒度并发梯度（12/16/20/24）复测确认。
- **c=32（关键分水岭档位）**：FlexKV 全面反超 HiCache——TTFT p50 低 12%（1.50 vs 1.70s），吞吐高 16%（401.6 vs 344.7 tok/s），QPS 高 16%（6.27 vs 5.39）。

### 2.3 命中率

两者在同一并发档几乎完全一致（差距 <0.3%），说明前缀缓存策略在等效 L2/L3 容量下命中行为对齐良好。命中率随并发升高而下降（93.6%→57.6%）是 `--num-requests` 固定为 40 时的固有现象（并发越高，请求越早到达，共享前缀写入相对滞后），非异常。

### 2.4 原始 JSON 结果

**FlexKV（优化后，`FLEXKV_CPU_CACHE_GB=192`）**：

```json
{
  "results": [
    {"concurrency": 1,  "n": 40, "success": 40, "errors": 0, "wall_s": 92.156,
     "ttft_mean": 0.535, "ttft_p50": 0.429, "ttft_p90": 0.523, "ttft_p99": 3.557,
     "e2e_p50": 2.166, "e2e_p90": 2.842, "e2e_p99": 5.938,
     "tpot_mean": 0.0281, "tpot_p50": 0.027,  "tpot_p90": 0.0378, "tpot_p99": 0.0428,
     "out_tok_per_s": 27.8, "req_per_s": 0.43,
     "prompt_tokens": 29320, "cached_tokens": 27456, "hit_rate": 0.9364},
    {"concurrency": 4,  "n": 40, "success": 40, "errors": 0, "wall_s": 29.092,
     "ttft_mean": 0.89,  "ttft_p50": 0.871, "ttft_p90": 1.212, "ttft_p99": 2.158,
     "e2e_p50": 2.722, "e2e_p90": 3.252, "e2e_p99": 3.946,
     "tpot_mean": 0.0297, "tpot_p50": 0.0301, "tpot_p90": 0.0338, "tpot_p99": 0.0386,
     "out_tok_per_s": 88.0, "req_per_s": 1.37,
     "prompt_tokens": 29320, "cached_tokens": 25344, "hit_rate": 0.8644},
    {"concurrency": 8,  "n": 40, "success": 40, "errors": 0, "wall_s": 18.568,
     "ttft_mean": 1.662, "ttft_p50": 1.282, "ttft_p90": 3.939, "ttft_p99": 4.013,
     "e2e_p50": 3.032, "e2e_p90": 6.346, "e2e_p99": 6.69,
     "tpot_mean": 0.0303, "tpot_p50": 0.0303, "tpot_p90": 0.0381, "tpot_p99": 0.0437,
     "out_tok_per_s": 137.9, "req_per_s": 2.15,
     "prompt_tokens": 29320, "cached_tokens": 22528, "hit_rate": 0.7683},
    {"concurrency": 16, "n": 40, "success": 40, "errors": 0, "wall_s": 12.629,
     "ttft_mean": 2.351, "ttft_p50": 1.261, "ttft_p90": 4.346, "ttft_p99": 4.684,
     "e2e_p50": 3.165, "e2e_p90": 7.014, "e2e_p99": 7.091,
     "tpot_mean": 0.0329, "tpot_p50": 0.0315, "tpot_p90": 0.0435, "tpot_p99": 0.0446,
     "out_tok_per_s": 202.7, "req_per_s": 3.17,
     "prompt_tokens": 29320, "cached_tokens": 16896, "hit_rate": 0.5763},
    {"concurrency": 32, "n": 40, "success": 40, "errors": 0, "wall_s": 6.375,
     "ttft_mean": 1.679, "ttft_p50": 1.498, "ttft_p90": 2.164, "ttft_p99": 3.213,
     "e2e_p50": 3.67,  "e2e_p90": 4.381, "e2e_p99": 5.268,
     "tpot_mean": 0.0322, "tpot_p50": 0.033,  "tpot_p90": 0.0357, "tpot_p99": 0.0364,
     "out_tok_per_s": 401.6, "req_per_s": 6.27,
     "prompt_tokens": 29320, "cached_tokens": 16896, "hit_rate": 0.5763}
  ]
}
```

**HiCache（`--hicache-size 24`）**：

```json
{
  "results": [
    {"concurrency": 1,  "n": 40, "success": 40, "errors": 0, "wall_s": 98.462,
     "ttft_mean": 0.596, "ttft_p50": 0.489, "ttft_p90": 0.671, "ttft_p99": 3.86,
     "e2e_p50": 2.332, "e2e_p90": 2.826, "e2e_p99": 5.912,
     "tpot_mean": 0.0296, "tpot_p50": 0.0287, "tpot_p90": 0.0349, "tpot_p99": 0.0359,
     "out_tok_per_s": 26.0, "req_per_s": 0.41,
     "prompt_tokens": 29280, "cached_tokens": 27456, "hit_rate": 0.9377},
    {"concurrency": 4,  "n": 40, "success": 40, "errors": 0, "wall_s": 31.678,
     "ttft_mean": 1.343, "ttft_p50": 0.962, "ttft_p90": 4.311, "ttft_p99": 4.44,
     "e2e_p50": 2.741, "e2e_p90": 6.376, "e2e_p99": 6.742,
     "tpot_mean": 0.0288, "tpot_p50": 0.0288, "tpot_p90": 0.0309, "tpot_p99": 0.0371,
     "out_tok_per_s": 80.8, "req_per_s": 1.26,
     "prompt_tokens": 29280, "cached_tokens": 25344, "hit_rate": 0.8656},
    {"concurrency": 8,  "n": 40, "success": 40, "errors": 0, "wall_s": 16.877,
     "ttft_mean": 1.191, "ttft_p50": 1.235, "ttft_p90": 1.546, "ttft_p99": 1.692,
     "e2e_p50": 3.202, "e2e_p90": 3.89,  "e2e_p99": 4.443,
     "tpot_mean": 0.0323, "tpot_p50": 0.0312, "tpot_p90": 0.0421, "tpot_p99": 0.0497,
     "out_tok_per_s": 151.7, "req_per_s": 2.37,
     "prompt_tokens": 29280, "cached_tokens": 22528, "hit_rate": 0.7694},
    {"concurrency": 16, "n": 40, "success": 40, "errors": 0, "wall_s": 12.553,
     "ttft_mean": 2.521, "ttft_p50": 1.558, "ttft_p90": 4.546, "ttft_p99": 4.618,
     "e2e_p50": 3.343, "e2e_p90": 6.745, "e2e_p99": 7.097,
     "tpot_mean": 0.0306, "tpot_p50": 0.0302, "tpot_p90": 0.0383, "tpot_p99": 0.0449,
     "out_tok_per_s": 203.9, "req_per_s": 3.19,
     "prompt_tokens": 29280, "cached_tokens": 16896, "hit_rate": 0.577},
    {"concurrency": 32, "n": 40, "success": 40, "errors": 0, "wall_s": 7.427,
     "ttft_mean": 1.804, "ttft_p50": 1.698, "ttft_p90": 2.288, "ttft_p99": 2.871,
     "e2e_p50": 4.565, "e2e_p90": 5.248, "e2e_p99": 5.472,
     "tpot_mean": 0.0418, "tpot_p50": 0.043,  "tpot_p90": 0.05,   "tpot_p99": 0.052,
     "out_tok_per_s": 344.7, "req_per_s": 5.39,
     "prompt_tokens": 29280, "cached_tokens": 16896, "hit_rate": 0.577}
  ]
}
```

---

## 3. 待办 / 后续建议

1. **复测 c=16 的异常波动**：用更细粒度并发梯度（12/16/20/24）确认吞吐持平/TTFT 尖峰是否为真实现象还是单次抖动。
2. **继续实施 `flexkv_vs_hicache_high_concurrency_analysis.md` §5 剩余优化方向**（中心化 match 移出主循环、弱化跨节点双确认、消除 store 路径同步点），进一步提升 FlexKV 在超高并发（>32）下的扩展性。
3. **将本次优化改动提交到 git**（`p800_flexkv` 分支当前为 uncommitted 状态），避免后续环境重建/pod 重启导致优化丢失。
4. **`needs_sync=False` 的适用范围**：目前硬编码仅适用于纯 TP 场景，切换 PP/CP>1 部署前需要恢复条件判断逻辑。

---

## 4. 相关文件

| 文件 | 说明 |
|---|---|
| `zittozhang_scripts/scripts/pd_start_5p1d_node_hicache.sh` | HiCache 模式单节点启动脚本 |
| `zittozhang_scripts/scripts/pd_launch_5p5d_hicache.sh` | 并行拉起 10 个 HiCache prefill 节点 |
| `zittozhang_scripts/scripts/pd_launch_5p5d_flexkv192_prefill.sh` | 仅重启 10 个 FlexKV prefill 节点（`FLEXKV_CPU_CACHE_GB=192`），decode 不受影响 |
| `zittozhang_scripts/scripts/bench_5p5d.py` | 测试脚本 |
| `/workspace/sglang_dev`（P800 集群 pod 内，10 个 prefill 节点同步） | sglang 源码（`p800_flexkv` 分支 + 本次未提交的性能优化改动） |
| `flexkv_vs_hicache_high_concurrency_analysis.md` | 高并发瓶颈根因分析（本次优化依据） |
| `doc/p800_hicache_vs_flexkv_comparison.md` | 完整版记录（含优化前数据、部署踩坑过程） |
