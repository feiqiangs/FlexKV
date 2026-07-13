# P800 5P5D：FlexKV 高并发下慢于 HiCache 的根因分析

> 本文档分析 P800 5P5D（5 prefill + 5 decode，每实例 tp16=2 节点）场景下，
> **FlexKV+mooncake-store** 作为 KVCache 管理器与 **sglang 原生 HiCache**（同样接 mooncake-store 做 L3）
> 在高并发时的性能分化：为什么并发升高（尤其 c=32）时 FlexKV 明显慢于 HiCache。
>
> - 对齐测试数据与结论见：`doc/p800_hicache_vs_flexkv_comparison.md`
> - 架构设计见：`doc/mooncake_store_p800_detailed_design.md`
> - 分析日期：2026-07-13

---

## 0. 结论先行（TL;DR）

两套方案在同一并发档的**命中率完全对齐**（c=32 都是 57.6%），说明性能差异**不来自缓存命中率**，而来自**请求处理 / 传输调度控制面的可扩展性**。

> **HiCache 是「每 TP rank 独立 + 后台线程/CUDA stream 全异步」的去中心化架构；
> FlexKV 是「整节点单例共享池 + rank0 中心化调度 + 每请求多次跨 TP16 集合通信 + 跨节点 TCP 完成确认」的中心化架构。**
>
> 高并发下 FlexKV 的这些串行化开销全部叠加在 sglang **调度主循环**里，随并发线性放大，
> 最终使**控制面（而非传输带宽）成为瓶颈**——表现为 c=32 时 TTFT p50 从 1.40s 暴涨到 5.07s、
> 吞吐不增反降（251.9 < c=16 的 260.0 tok/s）。

---

## 1. 关键测试数据回顾（c=16 / c=32 分水岭）

| 并发 | 方案 | 命中率 | TTFT p50 (s) | TTFT p90 (s) | out tok/s |
|---|---|---|---|---|---|
| 16 | FlexKV  | 57.6% | 1.40 | 1.68 | 260.0 |
| 16 | HiCache | 57.7% | 1.56 | 4.55 | 203.9 |
| 32 | FlexKV  | 57.6% | **5.07** | 5.75 | **251.9**（比 c=16 还降）|
| 32 | HiCache | 57.7% | **1.70** | 2.29 | **344.7**（近线性增长）|

- c≤8：两者基本打平，互有小幅胜负。
- **c=32 是分水岭**：FlexKV 的 TTFT p50 骤增、吞吐不增反降，出现明显拥塞/退化；HiCache 继续近线性扩展吞吐、TTFT 保持低位。

> 注：`--max-running-requests 16`，故 c=32 时恒有约一半请求在 waiting queue 排队。
> TTFT ≈ 排队等待 + prefill 计算 + KV 加载。两者排队上限相同，差异只可能来自**每请求的调度开销**。

---

## 2. 两套架构的关键差异

### 2.1 HiCache：去中心化、全异步

依据：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py`、
`managers/cache_controller.py`、`mem_cache/memory_pool_host.py`。

- **每个 TP rank 独立**：host 内存池按 rank 独立分配（`--hicache-size 24` × 8 rank = 192GB/节点），
  每 rank 拥有各自的 `HiRadixCache` + `HiCacheController`。
- **match 纯本地**：`match_prefix`（`hiradix_cache.py:781-816`）只是本 rank 的 radix 树遍历，
  轻量、**无任何跨 rank 通信**。
- **加载/写回全后台异步**：`prefetch` / `load` / `write_backup` 仅把 operation 入队后立即返回，
  实际 IO 在**独立的 load / write CUDA stream + 后台线程**上执行，与 GPU forward overlap；
  完成检测用 `finish_event.query()` **非阻塞轮询**（`check_hicache_events`）。
- **净效果**：调度主循环几乎不被 KV 管理拖慢，各 rank 完全并行、无中心化瓶颈、无跨节点确认。

### 2.2 FlexKV：中心化、多重同步

依据：`sglang/.../storage/flexkv/flexkv_connector.py`、`FlexKV/flexkv/kvtask.py`、
`FlexKV/flexkv/transfer_manager.py`、`sglang/.../storage/flexkv/flexkv_comm.py`。

本 5P5D prefill 实例为 **tp16 跨 2 节点**（每节点 8 卡），存在四处结构性串行化：

#### (1) 整节点单例 + rank0 中心化 match
- 整节点共享一份 CPU 池，由单个 `KVManager` / `KVTaskEngine` 管理，
  运行在 **sync_leader（TP rank0）的调度进程内**（`kvmanager.py:104-111` 非 server_client_mode → `KVTaskEngine`）。
- **只有 rank0 执行 `get_match` / `put_match`**（`flexkv_connector.py:344-359, 567`），
  在**主循环里同步**调用 `GlobalCacheEngine.get/put` 构建 transfer graph（`kvtask.py:189, 224`）。
  所有并发请求的前缀匹配 + 建图由 rank0 单线程串行处理。

#### (2) 每请求多次跨 TP16（跨节点）集合通信
rank0 算完后必须把结果广播给其余 15 个 rank，每请求的每个阶段都要走 `FlexKVComm` 的 gloo CPU 集合通信：

| 阶段 | 连接器方法 | 通信 | 代码位置 |
|---|---|---|---|
| 查询命中 | `get_new_hit_length` | `scatter` | `flexkv_connector.py:373-378` |
| 发起加载 | `start_load_kv` | `scatter_pp`（layerwise）/ `barrier`（非 layerwise）| `flexkv_connector.py:437-482` |
| 发起写回 | `start_store_kv` | `scatter_pp` | `flexkv_connector.py:516, 587` |
| 完成回收 | `check_completed_load/store_tasks` | `scatter` | `flexkv_connector.py:668-669` |

这些集合通信**全部串行阻塞在调度主循环**（`extended_radix_cache.py` 中 `match_prefix`→`get_new_hit_length`、
`ready_to_load_host_cache`→`start_load_kv`、`cache_finished_req`→`start_store_kv`、`check_kv_events`→完成检测均为主循环同步调用）。
跨 2 节点的 gloo CPU 通信延迟不低，且随并发内请求数线性累加。

#### (3) 跨节点 TCP 完成确认 + sleep 轮询
- `transfer_handles` 有 2 个：本地 `TransferManagerInterProcessHandle`（Pipe IPC）
  + 远程节点 B `TransferManagerMultiNodeHandle`（TCP ZMQ）（`kvtask.py:99-137`）。
- `required_completed_count = len(transfer_handles) = 2`（`kvtask.py:149`）：
  **每个 op/graph 必须等本地进程和远程节点 B「双方」都报告完成**才算完成（`kvtask.py:405-410`）。
- 远程节点 B 的完成经 TCP ZMQ 传回，master 侧 `MultiNodeHandle.wait` 仍是 `time.sleep(0.001)` 轮询
  （`transfer_manager.py:1094, 1172`）。高并发下大量小消息的序列化 + 轮询延迟累积。

#### (4) store 路径的隐藏同步点
- `start_store_kv` 里 `filtered.cpu()`（`flexkv_connector.py:592`）会触发 GPU 同步；
  graph 经 pickle over Pipe 提交（`transfer_manager.py:957`）。
- D2H（写回）与 H2D（加载）共享同一组 `GPUCPUTransferWorker` / copy engine，
  高并发下写回与加载互相争抢传输通道。

> 说明：本测试已启用 `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1`（`pd_start_5p1d_node.sh:54`），
> 故 GET 加载走 layerwise eventfd 异步路径（非 `start_load_kv` 中的同步 `wait` 阻塞分支），
> 已排除「同步等待 load 完成」这一路径。瓶颈集中在上述 (1)(2)(3)(4) 的控制面开销。

---

## 3. 与测试数据的对应

| 现象（c=32） | FlexKV | HiCache | 归因 |
|---|---|---|---|
| TTFT p50 | **5.07s** | 1.70s | 调度主循环被「rank0 串行 match + 每请求多次跨节点集合通信 + 跨节点 TCP 确认」拖慢，请求 admit 速率下降，排队时间暴涨 |
| 吞吐 out tok/s | 251.9（比 c=16 还降） | **344.7**（近线性增长） | FlexKV 控制面成瓶颈，GPU 等待调度而空转；HiCache 每 rank 并行 + 全异步，调度不拖后腿 |
| TPOT p50 | 36.3ms | 43.0ms | HiCache 略高，属 decode 侧正常竞争，不影响主结论 |

- **为什么 c≤8 时两者相当**：并发低时每轮循环请求少，FlexKV 的中心化/跨节点同步开销尚未累积成瓶颈。
- **为什么 c=16 是拐点、c=32 崩盘**：c=32 恒有约一半请求排队；FlexKV 每请求的固定调度开销
  （match + N 次集合通信 + IPC/TCP 确认）× 请求数 ≈ 超过调度循环预算，形成拥塞放大；
  HiCache 每请求调度开销接近零，故继续线性扩展。

---

## 4. 根因总结

FlexKV 慢的**不是数据传输带宽本身**（P800 CE 优化仍在），而是**控制面（调度 / 元数据 / 完成确认）的可扩展性**：

1. **中心化单例**：全节点靠 rank0 一个线程串行 match + 建图。
2. **跨 rank 同步放大**：每请求多次跨 TP16（跨 2 节点）gloo 集合通信，串行叠加在主循环。
3. **跨节点双确认 + sleep 轮询**：`required_completed_count=2` + TCP ZMQ + 1ms 轮询，延迟随并发累积。
4. **控制面阻塞主循环**：上述开销都在 sglang 调度主循环里同步执行，直接压低请求 admit 速率。

而 HiCache 的「每 rank 独立 + 后台线程/独立 CUDA stream 全异步 + 无跨节点确认」把这些开销从关键路径上彻底移除，因此高并发扩展性明显更好。

---

## 5. 可行的优化方向（供参考）

1. **把控制面移出调度主循环**：get/put match、launch、完成检测放到 FlexKV connector 内部的后台线程，
   主循环只做非阻塞入队/查询（对齐 HiCache 的 controller 线程模型）。
2. **削减每请求跨 rank 集合通信**：批量化 scatter（一轮循环合并多个请求的 hit_length/task_id 一次广播），
   或减少 barrier 次数。
3. **弱化跨节点双确认**：完成确认以本地节点为准 + 异步补偿远程，避免 `required_completed_count=2`
   的 TCP 轮询串行阻塞；把 `MultiNodeHandle.wait` 的 `time.sleep(0.001)` 轮询改为事件驱动
   （remote 侧已用 selector，master 侧仍是 sleep 轮询）。
4. **消除 store 路径同步点**：`filtered.cpu()` 的 D2H 拷贝改为在传输 worker 内异步完成，
   避免在主循环触发 GPU 同步。
5. **精确定位**：打开 `FLEXKV_D2H_PROFILE=1` 抓 `put_match_zmq` / `launch_zmq` 耗时，
   并在 c=32 下统计 rank0 主循环中 scatter / barrier 的累计占比，量化各阶段开销。

---

## 6. 关键代码位置索引

| 内容 | 文件 | 关键位置 |
|---|---|---|
| 连接器 match / load / store | `sglang/.../storage/flexkv/flexkv_connector.py` | `get_new_hit_length` L331-399、`start_load_kv` L406-484、`start_store_kv` L501-624、完成回收 L486-499/626-671 |
| 跨 rank 集合通信原语 | `sglang/.../storage/flexkv/flexkv_comm.py` | `FlexKVComm.scatter / scatter_pp / barrier / all_reduce_min` |
| 主循环调用点 | `sglang/.../mem_cache/extended_radix_cache.py` | `match_prefix` L119-175、`ready_to_load_host_cache` L257-270、`cache_finished_req` L272-338、`check_kv_events` L350-353 |
| 中心化任务引擎 | `FlexKV/flexkv/kvtask.py` | `KVTaskEngine.get_match/put_match` L581-689、`required_completed_count` L149、`_get_completed_ops` L399-418 |
| 传输管理 / 双 handle / TCP 轮询 | `FlexKV/flexkv/transfer_manager.py` | `transfer_handles` 构建 L99-137、`MultiNodeHandle.wait` L1159-1174、远程轮询 L1083-1098 |
| KVManager 单例 | `FlexKV/flexkv/kvmanager.py` | 非 server_client_mode → `KVTaskEngine` L104-111 |
| 启动参数（layerwise 开启） | `zittozhang_scripts/scripts/pd_start_5p1d_node.sh` | `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1` L54 |
| HiCache 对照实现 | `sglang/.../mem_cache/hiradix_cache.py`、`managers/cache_controller.py` | `match_prefix` L781-816、后台 load/write stream + 事件轮询 |
