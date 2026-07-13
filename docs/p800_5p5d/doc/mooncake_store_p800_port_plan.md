# 将 mooncake-store 分布式共享 KVCache 方案移植到 P800 版本 FlexKV — 实施方案

> 版本：v1.0  
> 日期：2026-06-30  
> 基线代码：`/FlexKV`（P800 昆仑适配版）、`/sglang`（P800 昆仑适配版）  
> 参考实现：`/mooncake/FlexKV`、`/mooncake/sglang`（NVIDIA + mooncake-store 分布式版）  
> 目标：在 **P800 版本 FlexKV** 上引入 mooncake-store 分布式 KVCache 池，实现跨节点 KVCache 共享。

---

## 0. 关键约束（务必遵守）

> **【硬约束 C1】P800 版本的 H2D / D2H 已通过 CE（Copy Engine）模式实现，且经过高度优化，移植时必须保留 P800 的这部分逻辑。**

- P800 的 GPU↔CPU 传输位于 `FlexKV/csrc/transfer.cu`，走 `cudaMemcpyAsync`（Copy Engine）的**自适应多路径**实现：
  - Path 0：两侧逻辑连续 → 单次大 memcpy；
  - Path 1：分段数 ≤ 阈值 → 逐段 memcpy；
  - Path 2：分段过多 → gather/scatter 流水线；
  - 含 MLA 专路、`compute_segments` 分段检测、sharded D2H（`block_stride != chunk_size`）边界保护、`FLEXKV_TRANSFER_FORCE_PATH` 等可调项。
- 对应 Python 侧的 `GPUCPUTransferWorker` / `tpGPUCPUTransferWorker`（`FlexKV/flexkv/transfer/worker.py`）以及 `use_ce_transfer_h2d / use_ce_transfer_d2h / transfer_num_cta_*` 参数链路，均属于 P800 已优化资产。
- **冲突处理原则**：mooncake 参考版在 `transfer.cu`、`layerwise.*`、`tp_transfer_thread_group.cpp` 以及 worker.py 中 H2D/D2H 相关逻辑若与 P800 版冲突，**一律保留 P800 版本逻辑**，仅在其之上「叠加」mooncake-store 的 CPU↔Remote（H2REMOTE / REMOTE2H）通路，不得覆盖或回退 P800 的 CE 优化。

其它约束：
- **C2**：不得整目录覆盖拷贝 mooncake 版代码（两版存在上游版本漂移），必须按 hunk cherry-pick。
- **C3**：mooncake-store 通路只允许操作 CPU host 内存与 RDMA，不得新增 GPU/CUDA/csrc 代码。
- **C4**：安全——RDMA 配置、master 地址等敏感参数从配置文件 / 环境变量注入，不硬编码。

---

## 1. 背景与目标

P800 版 FlexKV 当前的 remote 后端为 CFS/PCFS（以及 Mooncake **Transfer Engine** 点对点 RDMA，`mooncakeEngineWrapper.py`）。本方案要引入的是 mooncake-**store**（`mooncake.store.MooncakeDistributedStore`）——一个由多节点 CPU 内存组成、经 RDMA 零拷贝访问的**全局分布式 KVCache 池**，从而实现实例间 / 跨节点的 KVCache 复用。

**目标**：在 P800 FlexKV 上新增 `use_mooncake_store_backend` 后端，行为对齐 `/mooncake/FlexKV`，并保持 P800 既有 CE 优化与昆仑适配不受影响。

---

## 2. 现状分析与可行性结论

### 2.1 架构结论（已通过代码对比验证）

| 结论 | 证据 |
|---|---|
| mooncake-store 是 **CPU host 内存 + RDMA 网卡** 特性，**设备无关** | `MooncakeStoreTransferWorker` 仅操作 `self._cpu_buffer`，通过 SDK 的 `register_buffer/batch_put/batch_get` 做 RDMA |
| **不涉及任何 csrc/CUDA 改动** | P800 vs mooncake 的 `csrc` 差异约 1500 行，用 mooncake 关键字过滤命中 **0**，全为平台适配/版本漂移 |
| **sglang 侧对 mooncake-store 透明** | `/mooncake/sglang` 全量搜索 `use_mooncake_store_backend / mooncake_store_config` **零命中**；后端由 FlexKV 内部按配置切换 |
| 功能完全落在 **FlexKV Python 层** | 新增 `flexkv/external/` 模块 + 若干文件的 mooncake hunk |

### 2.2 数据流设计（分阶段 staging，复用 P800 既有 H2D/D2H）

传输类型图谱新增两个 **CPU 侧** 类型：
- `H2REMOTE (2→4)`：CPU buffer → mooncake store（写）
- `REMOTE2H (4→2)`：mooncake store → CPU buffer（读）

完整通路：

```
存储 (PUT) :  GPU  --[D2H, P800 CE 优化]-->  CPU buffer  --[H2REMOTE, 新增]-->  mooncake store
加载 (GET) :  mooncake store  --[REMOTE2H, 新增]-->  CPU buffer  --[H2D, P800 CE 优化]-->  GPU
```

> 其中 **D2H / H2D 完全复用 P800 已优化的 CE 路径（约束 C1）**；新增的 `MooncakeStoreTransferWorker` 只接管 CPU↔Remote 段，且只碰 host 内存。

---

## 3. 详细修改方案（FlexKV 侧）

> 以 P800 `/FlexKV` 为基线，按「先新增、后接入、再 rank 适配」顺序 cherry-pick。

### 3.1 纯新增（直接移植，低风险）

| 文件 | 改动 |
|---|---|
| `flexkv/external/__init__.py` | 新增，导出 3 个类 |
| `flexkv/external/mooncake_store_keys.py` | 新增，`PoolKind / PoolSpec / build_key`（key 后缀与 PP/layer-range 策略） |
| `flexkv/external/mooncake_store_utils.py` | 新增，`MooncakeStoreConfig / MooncakeStoreClient / MooncakeStoreCacheEngine` |
| `flexkv/transfer/worker.py` | **末尾追加** `MooncakeStoreTransferWorker`（self-contained，约 180 行）。**注意：仅追加新类，禁止改动既有 `GPUCPUTransferWorker / tpGPUCPUTransferWorker` 的 CE 逻辑（C1）** |

### 3.2 接入既有文件（按 hunk 合并，中风险）

| 文件 | mooncake-store 相关改动 | 注意事项 |
|---|---|---|
| `flexkv/common/config.py` | 新增配置项：`use_mooncake_store_backend`、`mooncake_store_config_path`、`mooncake_store_pp_rank/pp_size/node_layer_start/node_layer_end/total_layers`；新增 `enable_pool_specs()`；env 解析（`FLEXKV_USE_MOONCAKE_STORE_BACKEND` / `FLEXKV_MOONCAKE_STORE_CONFIG_PATH`） | config 夹杂上游版本漂移（`cp_size`/`nnodes_per_pp_rank` 等），**只挑 mooncake 字段**，不要带入漂移重构 |
| `flexkv/common/transfer.py` | `TransferOp` 新增 `mooncake_store_block_hashes` 字段；新增 `_merge_remote2h_ops`（REMOTE2H/H2REMOTE 合并，保留 mooncake hashes）；合并/校验逻辑放开 `REMOTE2H/H2REMOTE` 类型 | 合并逻辑需与 P800 现有 merge 流程兼容 |
| `flexkv/transfer/worker_op.py` | 透传 `mooncake_store_block_hashes`；mooncake 后端用 CPU block ids 计算指针 | 约 20 行，注意保留 P800 已有的 D2H profiling 字段差异 |
| `flexkv/transfer/transfer_engine.py` | 当 `use_mooncake_store_backend` 时，创建 `MooncakeStoreTransferWorker` 并注册到 `_worker_map[H2REMOTE/REMOTE2H]`（含 indexer 侧）；layerwise/合并路径透传 hashes | **H2D/D2H 的 worker 注册保持 P800 原状（C1）**，仅新增 H2REMOTE/REMOTE2H 分支 |
| `flexkv/cache/cache_engine.py` | `use_mooncake_store_backend` 时用 `MooncakeStoreCacheEngine` 作为 `remote_cache_engine`；match/insert 路径传 content-hash keys；`enable_kv_sharing` 与 mooncake 互斥处理 | — |
| `flexkv/storage/storage_engine.py` | mooncake 后端跳过 `RemoteAllocator`（remote 由 worker 直接管理，hot CPU buffer 复用，无需 staging 分配） | — |
| `flexkv/transfer_manager.py` | 计算本节点 CPU 池覆盖的层范围（`num_layers_on_node` / `node_layer_start/end` / `total_layers`），写回 cache_config 供 `build_key` 生成 PP key 后缀；mooncake 后端不创建 RemoteAllocator | **主要适配点**，见 §4.2 |

### 3.3 构建与依赖

| 文件 | 改动 |
|---|---|
| `FlexKV/install.sh` | 增加从源码构建 `mooncake-transfer-engine` 的步骤（`--mooncake-version`）、安装 `mooncake-store` Python SDK |
| `FlexKV/setup.py` | **无需改动**（mooncake-store 不引入 native 源文件）；保持 P800 现有 `csrc/*` 与 `-DCUDA_AVAILABLE` 链路 |

### 3.4 sglang 侧

- **基本无需改动**。mooncake-store 对 sglang 透明。
- 仅需保证 FlexKV 初始化时能拿到 `FLEXKV_USE_MOONCAKE_STORE_BACKEND` 与 `FLEXKV_MOONCAKE_STORE_CONFIG_PATH`（通过环境变量注入即可），无须改 `flexkv_connector.py` / `flexkv_comm.py` 的功能逻辑。
- **不要**把 mooncake 版 `flexkv_connector.py` 的差异搬过来——那 156 行差异是 sglang 上游版本漂移 + P800 专属 NUMA/profiling，与 mooncake-store 无关，强行合并会破坏 P800 行为。

---

## 4. 风险点与应对

### 4.1 R1：上游版本漂移（最高优先级）
- **现象**：`/mooncake/FlexKV` 基于更新的 FlexKV 上游（`config.py` 含 `cp_size`/`nnodes_per_pp_rank` 重构、PR #171 `num_layers_on_node`/`pp_start_layer`）。
- **应对**：禁止整目录覆盖（C2）；逐文件按 hunk 提取 mooncake 专属改动，apply 到 P800 基线；合并后逐文件过 lint + 单测。

### 4.2 R2：`transfer_manager.py` 的 rank/layer 模型差异（主要适配工作量）
- **现象**：P800 `transfer_manager.py` 无 `pp_start_layer`/`num_layers_on_node` 概念（命中 0；mooncake 命中 14）。mooncake 用其计算分布式 key 的 layer-range 后缀，避免跨节点 PP 拓扑串 key。
- **应对**：
  - 单机 / 单 PP：`build_key` 走「整模型层 → 无后缀」快路径，风险低，优先打通。
  - 跨节点 PP：需把 mooncake 的层范围计算适配到 P800 的 rank 模型（基于 P800 现有 `cp_size/nnodes_per_pp_rank`，config.py 命中 12，说明部分概念已存在）。建议分两阶段交付，跨节点 PP 作为第二阶段。

### 4.3 R3：`mooncake-transfer-engine` 在 P800 节点的编译与 RDMA 打通（最大不确定性）
- **现象**：唯一 native 外部依赖。基于 host 内存 + `libibverbs`，**不依赖 GPU**。
- **应对**：**优先做可行性预研**——确认 P800 节点 CPU 架构（x86/ARM）、RDMA 网卡（IB/RoCE）与 verbs 驱动就绪，能成功 build mooncake 及 Python SDK。`nic.txt` 可作为网卡信息参考。

### 4.4 R4：`cudaHostRegister` 兼容性
- **现象**：新 worker 用 `cudaHostRegister` 注册 pinned CPU buffer。
- **应对**：P800 `transfer.cu` 本身已在用 `cudaHostRegister`（存在 CUDA 兼容层），**沿用即可，无需额外适配**。

### 4.5 R5：与约束 C1 的冲突（CE 优化被回退）
- **现象**：合并 worker.py / transfer_engine.py / transfer.cu 时，mooncake 版可能携带不同的 H2D/D2H 实现。
- **应对**：合并前对 H2D/D2H 相关函数做「保护清单」，**只接受新增 H2REMOTE/REMOTE2H 相关 hunk**；对任何触及 `use_ce_transfer_*`、`GPUCPUTransferWorker`、`transfer.cu` CE 多路径的 hunk 一律丢弃并保留 P800 版。合并后用 D2H/H2D 正确性测试回归。

---

## 5. 实施步骤与里程碑

| 阶段 | 内容 | 预估 |
|---|---|---|
| M0 预研 | mooncake-transfer-engine 在 P800 编译 + RDMA 连通性验证（R3） | 1–3 人天 |
| M1 新增 | 移植 `flexkv/external/` + worker.py 追加新类（3.1） | ~1 人天 |
| M2 接入 | config / transfer / transfer_engine / cache_engine / storage_engine（3.2，遵守 C1） | 2–3 人天 |
| M3 单机打通 | 单机 / 单 PP 走无后缀快路径，端到端 PUT/GET 验证 | 1–2 人天 |
| M4 跨节点 PP | `transfer_manager.py` layer-range 适配（R2），跨节点 key 隔离验证 | 2–3 人天 |
| M5 联调回归 | 多机 KV 共享、与 P800 CE 路径回归（R5）、性能对比 | 2–3 人天 |

**总体估算：约 8–13 人天（约 1.5–2.5 周 / 单人）**，最大不确定性在 M0。

---

## 6. 验证方案

1. **功能**：参考 `/mooncake/FlexKV/tests/test_mooncake_store_integration.py` 移植单测；验证 `batch_put / batch_get / batch_exists` 与 `match()` prefix 命中。
2. **正确性**：跨实例写入后另一实例命中并加载，逐 block 校验 KV 内容一致。
3. **约束 C1 回归**：开启/关闭 mooncake 后端两种模式下，跑 P800 既有 D2H/H2D 正确性与性能用例，确认 CE 多路径行为与吞吐无回退。
4. **跨节点 PP**：PP=2/4 多拓扑下验证 key 后缀隔离正确（不同拓扑不串 cache）。
5. **性能**：对比 CFS/PCFS 后端，记录 H2REMOTE/REMOTE2H 带宽与端到端 TTFT 收益。

---

## 7. 附：关键开关与配置

- `FLEXKV_USE_MOONCAKE_STORE_BACKEND=1` 启用后端。
- `FLEXKV_MOONCAKE_STORE_CONFIG_PATH=/path/to/mooncake_store.json` 指定 store 配置（`master_addr / metadata_server / protocol / device_name / local_hostname / global_segment_size / ...`）。
- `FLEXKV_DEBUG_MOONCAKE_STORE=1` 打开传输调试日志。
- 启用 mooncake-store 后端时，CFS/PCFS remote 后端被替代，`enable_kv_sharing` 与之互斥。
