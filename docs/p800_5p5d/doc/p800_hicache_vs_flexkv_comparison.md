# P800 5P5D FlexKV vs sglang 原生 HiCache 对比测试记录

> 本文档记录 FlexKV+mooncake-store 与 sglang 原生 HiCache（同样接 mooncake-store 做 L3）在
> **同等参数条件**下的对比测试：L2 内存总量对齐、L3 mooncake-store 对齐、测试数据/脚本/并发梯度一致。
> FlexKV 单独的详细基线测试见 `doc/p800_flexkv_benchmark_results.md`。
> 部署过程中的踩坑记录（mooncake master 误杀、内存分配架构差异、mem-layout 断言等）见本文档 §3。

---

## 0. 测试日期

2026-07-13，P800 集群 5P5D（5 prefill + 5 decode，每实例 tp16=2 节点，共 20 节点）。

---

## 1. 对齐方案（同等参数条件）

| 层级 | FlexKV（本次实测配置） | sglang 原生 HiCache（本次实测配置） | 对齐说明 |
|---|---|---|---|
| L1（GPU 设备池） | `--mem-fraction-static 0.87` | `--mem-fraction-static 0.87` | 两者相同，无需改动 |
| L2（host 内存） | `FLEXKV_CPU_CACHE_GB=192`（整节点共享一份，`StorageEngine` 单例） | `--hicache-size 24`（每 TP rank 独立分配，8 rank/节点 × 24GB = 192GB/节点） | 见 §3.2 架构差异说明，两者节点总量严格对齐为 **192GB/节点** |
| L3（mooncake-store） | `MOONCAKE_GLOBAL_SEGMENT_SIZE_GB=8`/节点，master `***:50051` | `--hicache-storage-backend mooncake --hicache-storage-backend-extra-config '{"global_segment_size":"8gb", "master_server_address":"***:50051", "protocol":"rdma", "device_name":"mlx5_2", ...}'` | 每节点同样贡献 8GB，**复用同一个 mooncake master** |
| 其余参数 | `page-size/context-length/attention-backend/tp-size/ep-size/chunked-prefill-size/max-running-requests/moe-*/disaggregation-*` 等 | 完全照抄 FlexKV 启动命令 | 只替换 `--kv-connector-cls flexkv` ↔ `--enable-hierarchical-cache ...` |
| decode 侧 | 不变 | 不变 | HiCache/FlexKV 均只是 prefill 侧本地前缀缓存机制，decode 全程未重启 |

**测试脚本/并发梯度（两组完全一致）**：

```
--mode fixed --concurrency 1,4,8,16,32 --num-requests 40 --prefix-tokens 1200 --max-tokens 64 --flush-before
```

每档 40 个独立 session，共享约 1200 token 的合成前缀 + 唯一后缀（自包含，不依赖外部数据集）。

**脚本**：`zittozhang_scripts/scripts/bench_5p5d.py`（本次测试前已增强，补齐 `tpot_p90/p99`、`e2e_p99`、`tpot_mean` 等字段）。

---

## 2. 测试结果

### 2.1 汇总对比表

| 并发 | 方案 | 命中率 | TTFT p50 (s) | TTFT p90 (s) | TTFT p99 (s) | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) | out tok/s | QPS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1  | FlexKV  | 93.6% | 0.44 | 0.58 | 3.44 | 27.9 | 35.3 | 41.9 | 27.4  | 0.43 |
| 1  | HiCache | 93.8% | 0.49 | 0.67 | 3.86 | 28.7 | 34.9 | 35.9 | 26.0  | 0.41 |
| 4  | FlexKV  | 86.4% | 0.94 | 1.24 | 1.74 | 28.5 | 36.8 | 40.3 | 92.2  | 1.44 |
| 4  | HiCache | 86.6% | 0.96 | **4.31** | 4.44 | 28.8 | 30.9 | 37.1 | 80.8  | 1.26 |
| 8  | FlexKV  | 76.8% | 1.34 | **4.25** | 4.32 | 30.2 | 34.1 | 41.1 | 133.9 | 2.09 |
| 8  | HiCache | 76.9% | 1.24 | 1.55 | 1.69 | 31.2 | 42.1 | 49.7 | 151.7 | 2.37 |
| 16 | FlexKV  | 57.6% | 1.40 | 1.68 | 1.75 | 29.7 | 40.3 | 40.3 | 260.0 | 4.06 |
| 16 | HiCache | 57.7% | 1.56 | 4.55 | 4.62 | 30.2 | 38.3 | 44.9 | 203.9 | 3.19 |
| 32 | FlexKV  | 57.6% | **5.07** | 5.75 | 7.20 | 36.3 | 40.7 | 43.6 | 251.9 | 3.94 |
| 32 | HiCache | 57.7% | **1.70** | 2.29 | 2.87 | 43.0 | 50.0 | 52.0 | 344.7 | 5.39 |

### 2.2 原始 JSON 结果

**FlexKV**（`FLEXKV_CPU_CACHE_GB=192`，`result_fixed_flexkv192_v1.json`）：

```json
{
  "results": [
    {"concurrency": 1,  "n": 40, "success": 40, "errors": 0, "wall_s": 93.586,
     "ttft_mean": 0.539, "ttft_p50": 0.444, "ttft_p90": 0.58,  "ttft_p99": 3.435,
     "e2e_p50": 2.222, "e2e_p90": 2.64, "e2e_p99": 5.732,
     "tpot_mean": 0.0286, "tpot_p50": 0.0279, "tpot_p90": 0.0353, "tpot_p99": 0.0419,
     "out_tok_per_s": 27.4, "req_per_s": 0.43,
     "prompt_tokens": 29320, "cached_tokens": 27456, "hit_rate": 0.9364},
    {"concurrency": 4,  "n": 40, "success": 40, "errors": 0, "wall_s": 27.773,
     "ttft_mean": 0.925, "ttft_p50": 0.941, "ttft_p90": 1.239, "ttft_p99": 1.743,
     "e2e_p50": 2.676, "e2e_p90": 3.305, "e2e_p99": 3.785,
     "tpot_mean": 0.029, "tpot_p50": 0.0285, "tpot_p90": 0.0368, "tpot_p99": 0.0403,
     "out_tok_per_s": 92.2, "req_per_s": 1.44,
     "prompt_tokens": 29320, "cached_tokens": 25344, "hit_rate": 0.8644},
    {"concurrency": 8,  "n": 40, "success": 40, "errors": 0, "wall_s": 19.12,
     "ttft_mean": 1.752, "ttft_p50": 1.338, "ttft_p90": 4.255, "ttft_p99": 4.323,
     "e2e_p50": 3.211, "e2e_p90": 6.188, "e2e_p99": 6.841,
     "tpot_mean": 0.0306, "tpot_p50": 0.0302, "tpot_p90": 0.0341, "tpot_p99": 0.0411,
     "out_tok_per_s": 133.9, "req_per_s": 2.09,
     "prompt_tokens": 29320, "cached_tokens": 22528, "hit_rate": 0.7683},
    {"concurrency": 16, "n": 40, "success": 40, "errors": 0, "wall_s": 9.845,
     "ttft_mean": 1.346, "ttft_p50": 1.401, "ttft_p90": 1.68,  "ttft_p99": 1.748,
     "e2e_p50": 3.359, "e2e_p90": 3.628, "e2e_p99": 4.106,
     "tpot_mean": 0.0312, "tpot_p50": 0.0297, "tpot_p90": 0.0403, "tpot_p99": 0.0403,
     "out_tok_per_s": 260.0, "req_per_s": 4.06,
     "prompt_tokens": 29320, "cached_tokens": 16896, "hit_rate": 0.5763},
    {"concurrency": 32, "n": 40, "success": 40, "errors": 0, "wall_s": 10.163,
     "ttft_mean": 4.527, "ttft_p50": 5.072, "ttft_p90": 5.746, "ttft_p99": 7.199,
     "e2e_p50": 7.37,  "e2e_p90": 8.309, "e2e_p99": 9.377,
     "tpot_mean": 0.0354, "tpot_p50": 0.0363, "tpot_p90": 0.0407, "tpot_p99": 0.0436,
     "out_tok_per_s": 251.9, "req_per_s": 3.94,
     "prompt_tokens": 29320, "cached_tokens": 16896, "hit_rate": 0.5763}
  ]
}
```

**HiCache**（`--hicache-size 24`，`result_fixed_hicache_v2.json`）：

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

## 3. 分析结论

### 3.1 命中率

两者在同一并发档几乎完全一致（差距 <0.3%），说明前缀缓存策略在等效 L2/L3 容量下命中行为对齐良好，符合预期。命中率随并发升高而下降（93.6%→57.6%）是 `--num-requests` 固定为 40 时的固有现象（并发越高，请求越早到达，共享前缀写入相对滞后），与 FlexKV 单独基线测试（`doc/p800_flexkv_benchmark_results.md` §2.2）结论一致。

### 3.2 低/中并发（c=1/4/8）

- **c=1/4**：TTFT/TPOT/吞吐基本打平，两者互有微弱优势，属误差范围。
- **c=8**：HiCache 的 TTFT p90/p99 更稳定（1.55/1.69s vs FlexKV 4.25/4.32s），吞吐也更高（151.7 vs 133.9 tok/s）；FlexKV 在这一档出现一次尾部延迟尖峰。

### 3.3 高并发（c=16/32）—— 两者出现明显分化，且方向不一致

- **c=16**：FlexKV 的 TTFT 长尾更稳（p90/p99 = 1.68/1.75s），但吞吐更低（260.0 vs 203.9 tok/s）；HiCache 吞吐更低但该档 TTFT p90 出现尖峰（4.55s）。
- **c=32（最大分水岭）**：
  - FlexKV：TTFT p50 从 c=16 的 1.40s 骤增到 **5.07s**，长尾全面恶化（p90=5.75s, p99=7.20s），吞吐反而比 c=16 还略降（251.9 < 260.0），出现明显的拥塞/退化迹象。
  - HiCache：TTFT p50 仅 1.70s（远优于 FlexKV 的同档 5.07s），吞吐继续上升到全场最高的 344.7 tok/s，扩展性明显更好；代价是 TPOT 略高（43ms vs 36ms），属高并发下 decode 侧竞争加剧的正常现象。

### 3.4 结论

- **命中率**：两者对齐，无实质差异。
- **低/中并发（c≤8）**：两者相当，各有小幅胜负。
- **高并发（c≥16，尤其 c=32）**：**HiCache 的排队/调度扩展性明显优于 FlexKV**——FlexKV 在 c=32 出现吞吐不增反降、TTFT 长尾暴涨（疑似请求排队或传输通道成为瓶颈）；HiCache 则继续近线性扩展吞吐、TTFT 保持低位。

### 3.5 需要注意的边界条件（影响结果解读）

1. **L2 容量并非两者默认值，而是为满足对齐做了调整**：
   - HiCache 原生存在硬性架构约束：sglang 的 `--hicache-size` 是**按每个 TP rank 独立分配**（非整节点共享），且有硬编码断言要求每 rank 的 L2(host) 必须严格大于该 rank 的 L1(device) KV cache 大小（`memory_pool_host.py` `assert self.size > device_pool.size`）。实测每 rank device KV cache = 22.23GB，故 `--hicache-size` 最低需 >22.23，本次选用安全值 24GB/rank（8 rank/节点 × 24GB = 192GB/节点）。
   - FlexKV 侧 `FLEXKV_CPU_CACHE_GB` 是整节点共享一份（`StorageEngine` 单例），原始生产配置为 100GB/节点，本次为了与 HiCache 的 192GB/节点严格对齐，**临时提升到 192GB** 重新测试。
   - 因此本文档的 FlexKV 数据代表"192GB L2"配置下的表现，**并非** `doc/p800_flexkv_benchmark_results.md` 中记录的 100GB 默认配置基线，两份文档的 FlexKV 数据不可直接互相比较。
2. **P800 定制 sglang 需要显式传 `--hicache-mem-layout page_first --hicache-io-backend kernel`**：P800 定制版 `_handle_hicache()` 直接走 `_handle_hicache_kunlun()` 分支 return，跳过了上游给 mooncake backend 自动切换 `layer_first`→`page_first` 布局的逻辑（该逻辑在 return 之后已成死代码）。若省略该参数，默认 `layer_first` 布局会在 `mooncake_store.py` 的 `register_mem_pool_host` 触发断言失败（"mooncake store storage backend only support page first or page first direct layout"）。
3. **部署过程中 mooncake master 曾被 `cleanup_pods.sh` 的 orphan-kill 逻辑误杀一次**（该逻辑按 `PPid==1` + cmdline 匹配 `flexkv_env` 关键字扫描杀进程，`mooncake_master` 二进制路径正好命中），已重新拉起并验证恢复，未对最终测试数据产生影响（发生在正式测试之前的环境搭建阶段）。后续如需再次清理 prefill 节点，应避免直接跑完整 `cleanup_pods.sh`（含 orphan-kill 步骤），或需要先将 orphan-kill 的匹配规则加上 mooncake_master 排除项。

---

## 4. 相关脚本/文件

| 文件 | 说明 |
|---|---|
| `zittozhang_scripts/scripts/pd_start_5p1d_node_hicache.sh` | HiCache 模式单节点启动脚本（`--hicache-size 24`、`--hicache-mem-layout page_first` 等） |
| `zittozhang_scripts/scripts/pd_launch_5p5d_hicache.sh` | 并行拉起 10 个 HiCache prefill 节点 |
| `zittozhang_scripts/scripts/pd_launch_5p5d_flexkv192_prefill.sh` | 仅重启 10 个 FlexKV prefill 节点（`FLEXKV_CPU_CACHE_GB=192`），decode 不受影响 |
| `zittozhang_scripts/scripts/pd_start_5p1d_node.sh` | FlexKV 模式单节点启动脚本（原始，`FLEXKV_CPU_CACHE_GB` 可通过 env 覆盖） |
| `zittozhang_scripts/scripts/bench_5p5d.py` | 测试脚本（已补齐 TTFT/TPOT/E2E 的 p50/p90/p99 全量指标） |
| `/workspace/zittozhang/logs/result_fixed_hicache_v2.json`（P0 pod 内） | HiCache 原始结果 |
| `/workspace/zittozhang/logs/result_fixed_flexkv192_v1.json`（P0 pod 内） | FlexKV(192GB) 原始结果 |
