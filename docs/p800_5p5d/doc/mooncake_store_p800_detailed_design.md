# mooncake-store 分布式共享 KVCache 移植到 P800 版 FlexKV — 详细设计文档

> **【约束与注意事项】如果遇到任何不确定的地方，不要自作主张，要随时问我。**
>
> 本文档涉及 P800 昆仑适配版的高度优化逻辑与 mooncake 上游版本漂移，任何拿不准的合并取舍、接口语义、字段含义、rank/layer 拓扑映射、RDMA/编译环境差异，都必须先与我确认后再动手，禁止凭猜测改动代码或覆盖既有逻辑。

---

> 版本：v1.0
> 日期：2026-06-30
> 上游实施方案：`doc/mooncake_store_p800_port_plan.md`（本文档是其落地的**详细设计**）
> 基线代码：`/FlexKV`（P800 昆仑适配版）、`/sglang`（P800 昆仑适配版）
> 参考实现：`/mooncake/FlexKV`、`/mooncake/sglang`（NVIDIA + mooncake-store 分布式版）

---

## 0. 强约束（继承自实施方案，务必遵守）

| 编号 | 约束 | 说明 |
|---|---|---|
| **C1** | **保留 P800 的 H2D/D2H CE（Copy Engine）优化** | `FlexKV/csrc/transfer.cu` 的自适应多路径、`GPUCPUTransferWorker`/`tpGPUCPUTransferWorker` 的 `use_ce_transfer_*` / `transfer_num_cta_*` 链路，以及 P800 专属的 `_safe_checksum`、带 `FLEXKV_DETAIL_TIMING` 计时的 `launch_transfer`，**一律保留，禁止被 mooncake 上游版覆盖**。 |
| **C2** | **禁止整目录覆盖拷贝** | 两版存在上游版本漂移，必须逐文件按 hunk cherry-pick。 |
| **C3** | **mooncake-store 通路只碰 CPU host 内存 + RDMA** | 不得新增任何 GPU/CUDA/csrc 代码。 |
| **C4** | **安全** | RDMA 配置、master 地址等敏感参数从配置文件 / 环境变量注入，**禁止硬编码**（见 §9）。 |

> 详细约束背景见 `doc/mooncake_store_p800_port_plan.md` §0。本文档在该方案基础上，给出**可直接编码落地的接口级 / 数据流级设计**。

---

## 1. 设计目标与范围

### 1.1 目标
在 P800 版 FlexKV 中新增 `use_mooncake_store_backend` 后端，引入 `mooncake.store.MooncakeDistributedStore` 作为**跨节点全局分布式 KVCache 池**（CPU 内存 + RDMA 零拷贝），实现实例间 / 跨节点 KVCache 复用，行为对齐 `/mooncake/FlexKV`，且不影响 P800 既有 CE 优化与昆仑适配。

### 1.2 范围
- **改动落在 FlexKV Python 层**：新增 `flexkv/external/` 包，叠加若干文件的 mooncake hunk。
- **sglang 侧透明**：仅靠环境变量注入开关，不改功能逻辑。
- **不改 csrc / CUDA**。

### 1.3 非目标
- 不替换或重写 P800 的 D2H/H2D CE 路径。
- 不引入 mooncake 上游的 `flexkv_connector.py` 漂移差异。
- 跨节点 PP 的完整支持作为第二阶段（见 §6.2）。

---

## 2. 总体架构

### 2.1 设备/传输类型图谱（device id）

P800 现有类型 + 新增两个 **CPU↔Remote** 类型：

| device id | 设备 |
|---|---|
| 1 | GPU |
| 2 | CPU（hot buffer） |
| 3 | SSD |
| **4** | **REMOTE（mooncake store）** |
| 5/6 | PEER CPU/SSD |

新增传输类型（`flexkv/common/transfer.py::TransferType`）：

| 类型 | (src,dst) | 语义 |
|---|---|---|
| `H2REMOTE` | (2,4) | CPU buffer → mooncake store（写 / PUT） |
| `REMOTE2H` | (4,2) | mooncake store → CPU buffer（读 / GET） |

### 2.2 分阶段 staging 数据流（复用 P800 CE）

```
PUT :  GPU --[D2H, P800 CE 优化]--> CPU hot buffer --[H2REMOTE, 新增]--> mooncake store
GET :  mooncake store --[REMOTE2H, 新增]--> CPU hot buffer --[H2D, P800 CE 优化]--> GPU
```

- **D2H / H2D 完全复用 P800 CE 路径**（约束 C1）。
- 新增的 `MooncakeStoreTransferWorker` **只接管 CPU↔Remote 段，且只操作已注册的 host 内存指针**。
- **关键设计：零额外 staging**。mooncake 后端直接把 **hot CPU buffer** 注册给 store（`register_buffer`），PUT/GET 时按 `block_id` 计算指针偏移，对该 buffer 原地 RDMA 读写，无需 `RemoteAllocator` 分配独立 staging 区。

### 2.3 模块分层

```
┌────────────────────────────────────────────────────────────┐
│ sglang (透明, 仅注入 env)                                    │
├────────────────────────────────────────────────────────────┤
│ FlexKV 控制面                                                │
│  KVManager → TransferManager → {CacheEngine, TransferEngine} │
│                                   │            │             │
│              remote_cache_engine ─┘            │             │
│              = MooncakeStoreCacheEngine        │             │
│                                   worker_map[H2REMOTE/REMOTE2H]
│                                          = MooncakeStoreTransferWorker
├────────────────────────────────────────────────────────────┤
│ flexkv/external/ (新增适配层)                                 │
│  mooncake_store_keys.py   : PoolKind/PoolSpec/build_key      │
│  mooncake_store_utils.py  : Config/Client/CacheEngine        │
├────────────────────────────────────────────────────────────┤
│ mooncake.store.MooncakeDistributedStore (外部 Python SDK)     │
│  + libmooncake-transfer-engine (native, host 内存 + libibverbs)│
└────────────────────────────────────────────────────────────┘
```

---

## 3. 新增模块详细设计（`flexkv/external/`）

> 这三个文件为**纯新增、低风险**，可直接从 `/mooncake/FlexKV/flexkv/external/` 移植，需逐文件 review 与 P800 import 路径一致性。

### 3.1 `__init__.py`
- 导出 `MooncakeStoreConfig / MooncakeStoreClient / MooncakeStoreCacheEngine`。
- 用模块级 `__getattr__` 做**惰性导入**：不使用本后端的进程不付出 `import torch/requests/mooncake` 成本。

### 3.2 `mooncake_store_keys.py` — Key 规则（单一事实源）

#### 3.2.1 数据结构
```python
class PoolKind(str, Enum):
    KV      = "FlexKV"          # 主 KV-cache block（默认主流量）
    INDEXER = "FlexKV_indexer"  # DSA indexer 边车（每主 block 一个，同 block-id 命名空间）
    SWA     = "FlexKV_swa"      # 滑窗注意力边车（已定义，当前未启用）

@dataclass(frozen=True)
class PoolSpec:
    kind: PoolKind
    required_for_hit: bool = True   # 是否参与"全部命中"判定
```

#### 3.2.2 `build_key(...)` — Key 生成核心
```python
build_key(block_hash, kind,
          pp_rank=0, pp_size=1,
          node_layer_start=0, node_layer_end=0, total_layers=0) -> str
```

| Case | 条件 | 返回 |
|---|---|---|
| 基础 | — | `"{block_hash}_{kind.value}"`，如 `"<hash>_FlexKV"` |
| **Case 1 整模型节点（无后缀）** | `total_layers>0` 且 `node_layer_end-node_layer_start == total_layers` | `base`（**不加后缀**） |
| **Case 2 跨节点 PP** | 未命中 Case 1 且 `pp_size>1` | `"{base}_pp_rank_{pp_rank}_of_{pp_size}"` |
| 兼容旧调用 | `total_layers==0` | 跳过 Case 1，落入 Case 2 或返回 `base` |

**设计要点**：
- 是否"单节点/全模型"由 **节点 CPU 池覆盖的 layer-range** 决定，而非直接看 `pp_size`——因为单节点 PP>1 部署其每节点 CPU 池仍覆盖全模型（与 PP=1 落盘 block 按位一致），可跨拓扑共享 key 互相命中。
- 跨节点 PP 各节点层切片长度不同，落盘 block 按位不兼容，必须用 `_pp_rank_i_of_N` 后缀隔离，**避免 alias 命中错误数据**。

> ⚠️ **不确定点（须与我确认）**：P800 的 rank 模型字段（`cp_size`/`nnodes_per_pp_rank`）与 mooncake 的 `pp_start_layer`/`num_layers_on_node` 不完全一致，layer-range 的计算映射务必确认后再落（见 §6.2、§7.2）。

### 3.3 `mooncake_store_utils.py` — 适配层主体

#### 3.3.1 `MooncakeStoreConfig`（dataclass）
从 JSON 文件加载（`from_file(cache_config, override_global_segment_size=None)`）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `master_addr` | `""` | mooncake master/etcd 端点，如 `192.168.1.1:2379` |
| `metadata_server` | `"P2PHANDSHAKE"` | 元数据服务类型串 |
| `protocol` | `"rdma"` | `rdma` 或 `tcp` |
| `device_name` | `""` | RDMA 设备名（如 `mlx5_0`），空=自动选 |
| `local_hostname` | `""` | 本地 IP/hostname，用于 buffer 注册 |
| `global_segment_size` | `256 GiB` | 注册到 store 的内存段大小（JSON 中单位 GB，`from_file` 内 `*1024**3`） |
| `enable_ssd_offload` / `ssd_offload_path` | `False`/`None` | SSD 卸载（SDK setup 暂注释，未启用） |
| `master_metrics_port` | `9003` | master metrics 端口（健康检查用） |

- `override_global_segment_size=0` → 创建**纯客户端**（pure-client）实例，不贡献内存段。用于 sidecar（如 indexer）worker。

#### 3.3.2 `MooncakeStoreClient`
对 `MooncakeDistributedStore` 的薄封装；**lazy setup**，可跨进程 pickle config。

| 方法 | 行为 |
|---|---|
| `setup()` | 进程内首用时 `import mooncake.store`，调 `store.setup(local_hostname, metadata_server, global_segment_size, local_buffer_size, protocol, device_name, master_addr)`；`query_only` 时 protocol=`rpc_only`、两个 size=0 |
| `check_server()` | 轮询 `http://{master_ip}:{master_metrics_port}/get_all_segments`，超时 `SETUP_TIMEOUT=600s`；pure-client 须等到已有 segment 出现 |
| `warm_up()` | 写 4KB warmup key（最多重试 10 次），再 `is_exist`/`get` 自校验，规避 Transfer Engine 启动竞态 |
| `register_buffer(tensor_or_ptr, size=0)` | 注册 pinned CPU buffer（tensor 自动取 `data_ptr/size`），失败抛错 |
| `unregister_buffer(...)` | 反注册 |
| `batch_put(keys, ptrs, sizes)` | 先 `batch_exists_impl` 过滤已存在 key（去重幂等），再 `zero_copy_put_impl`(=`store.batch_put_from`)，返回 `List[bool]`（`ret==0` 为成功） |
| `batch_get(keys, ptrs, sizes)` | `zero_copy_get_impl`(=`store.batch_get_into`)，返回 `List[bool]`（`ret>0` 字节数为成功） |
| `batch_exists(keys)` | 返回**最长存在前缀长度**（首个不存在处截断） |
| `batch_exists_impl(keys)` | 返回每 key 原始状态码列表（1 存在/0 缺失/-1 错误） |
| `clear()` | `store.remove_all()` |

**安全/幂等设计**：`batch_put` 写前去重；put 成功判据 `ret==0`，get 成功判据 `ret>0`。

#### 3.3.3 `MooncakeStoreCacheEngine`（作为 `remote_cache_engine`）
对齐 `CacheEngineAccel` 的 `match()`/`insert()` 接口，**只做存在性查询，不管理本地 block id**。

- `__init__(cache_config)`：读 `mooncake_store_pp_rank/pp_size/node_layer_start/node_layer_end/total_layers`、`tokens_per_block`；`pool_specs = cache_config.enable_pool_specs()`，筛 `required_for_hit` 得 `hit_pool_specs`；创建 `MooncakeStoreClient(query_only=True)`（纯查询，不贡献段）。
- `MATCHED_POS = "global"`：哨兵值，GET 路径据此分支。
- **`match(sequence_meta) -> MatchResultAccel`**：
  - 对每个 block_hash、每个 `hit_pool_spec` 用 `build_key` 生成 key，平铺成一个 list：`[pool0_blk0, pool0_blk1, ..., pool1_blk0, ...]`，单次 RPC。
  - **单池**（仅 KV）：`batch_exists` 直接取最长前缀。
  - **多池**（KV+indexer/...）：`batch_exists_impl` 取原始码，逐 block 要求**所有 required 池同时存在**才计入命中前缀。
  - 返回 `MatchResultAccel(num_matched_blocks=matched_length, physical_blocks=np.arange(matched_length), matched_pos="global", ...)`（physical_blocks 为占位，实际用 key 寻址）。
- **`insert(...)`** / `reset/lock/unlock/set_ready/insert_and_publish/take/recycle`：**no-op**（真正写入由 transfer worker 完成；索引由 store 自管）。`take` 返回全 0 占位。

---

## 4. 传输 Worker 设计（`flexkv/transfer/worker.py`）

> **【C1 保护清单】** 移植时**只在文件末尾追加** `MooncakeStoreTransferWorker`，**禁止改动** `GPUCPUTransferWorker` / `tpGPUCPUTransferWorker` 的以下内容：
> - `use_ce_transfer_h2d/d2h`、`transfer_num_cta_h2d/d2h` 参数与 `_transfer_impl` 的方向分支
> - C++ 调用 `transfer_kv_blocks(...)` / `TPTransferThreadGroup.tp_group_transfer(...)`
> - P800 专属的 `_safe_checksum`（`FLEXKV_CHECKSUM_DEBUG`）与带 `FLEXKV_DETAIL_TIMING` 计时的 `launch_transfer`（`[PATCH 06-10]`）
>
> ⚠️ mooncake 上游版的 `GPUCPUTransferWorker`/`tpGPUCPUTransferWorker` 还携带 PP 漂移（新增 `start_layer_id`）。**该漂移与 mooncake-store 无关**，是否需要 PP 支持须与我确认后再决定是否引入（见 §6.2）。

### 4.1 `MooncakeStoreTransferWorker(TransferWorkerBase)`（约 180 行，self-contained）

#### 4.1.1 `__init__`
```python
__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor,
         cpu_blocks, cpu_kv_layout, dtype, cache_config,
         pool_kind=PoolKind.KV, override_global_segment_size=None)
```
关键步骤：
1. 从 `cache_config` 读 PP/layer-range（`mooncake_store_pp_rank/pp_size/node_layer_start/node_layer_end/total_layers`），默认值保证 `pp_size==1` 单机调用零影响。
2. `self.suffix_str = pool_kind.value`，`self.pool_kind = pool_kind`。
3. `materialize_worker_tensor(cpu_blocks)` 取得 hot CPU tensor → **`cudaHostRegister(cpu_blocks)`** 注册 pinned（**沿用 P800 既有 `cudaHostRegister`，符合 R4，无需额外适配**）。
4. 断言 `cpu_kv_layout.type == BLOCKFIRST`；记录 `num_layers/num_cpu_blocks/block_size/is_mla`。
5. `MooncakeStoreConfig.from_file(cache_config, override_global_segment_size)` → `MooncakeStoreClient(store_config)` → `register_buffer(self._cpu_buffer)` 把整块 hot CPU buffer 注册给 store。

#### 4.1.2 传输实现
- `_preprocess(transfer_op)`：
  - REMOTE2H 用 `dst_block_ids`（填 CPU），H2REMOTE 用 `src_block_ids`（读 CPU）。
  - `block_size_bytes = elements_per_block * dtype.itemsize`。
  - 逐 block：`cpu_ptr = base_ptr + blk_id * block_size_bytes`；`key = build_key(mooncake_store_block_hashes[i], pool_kind, pp_*, node_layer_*, total_layers)`。
  - 返回 `(cpu_ptrs, block_sizes, keys)`。
  - **断言 `transfer_op.mooncake_store_block_hashes is not None`**。
- `_transfer_impl(cpu_ptrs, block_sizes, keys, transfer_type)`：
  - `H2REMOTE` → `mooncake_client.batch_put(keys, cpu_ptrs, block_sizes)`
  - `REMOTE2H` → `mooncake_client.batch_get(keys, cpu_ptrs, block_sizes)`
  - 其它类型抛 `ValueError`。
  - `FLEXKV_DEBUG_MOONCAKE_STORE` 控制 ok/fail 日志。
- `launch_transfer(transfer_op)`：`_preprocess` → 计时 → `_transfer_impl` → `_log_transfer_performance`（**这是 mooncake worker 自己的精简版，不影响 P800 的 GPUCPU worker 计时逻辑**）。

#### 4.1.3 寻址契约
> Key 格式 `"{block_hash}_{kind.value}"`：**一个 block 一个 key**，所有 layer 作为单一连续 slab 存于同一 key 下。这要求 hot CPU layout 为 `BLOCKFIRST` 且 block 内 layer 连续。

---

## 5. 既有文件接入设计（按 hunk 合并，中风险）

> 顺序：先新增（§3/§4）→ 接入配置/数据结构 → 接入引擎 → rank/layer 适配（§6.2）。每文件合并后过 lint + 单测。

### 5.1 `flexkv/common/config.py`
- `CacheConfig` 新增字段：`use_mooncake_store_backend`、`mooncake_store_config_path`、`mooncake_store_pp_rank/pp_size/node_layer_start/node_layer_end/total_layers`（默认安全，PP=1 单机无后缀）。
- `__post_init__`：解析 `FLEXKV_USE_MOONCAKE_STORE_BACKEND`、缺省时从 `FLEXKV_MOONCAKE_STORE_CONFIG_PATH` 取路径，二者皆空且启用后端则报错。
- 新增 `enable_pool_specs()`：`[PoolSpec(KV)]`，若 `indexer is not None` 追加 `PoolSpec(INDEXER)`（确定性顺序，供 worker 创建 / match / key 构建共用同一列表）。
- 派生标志：`enable_remote = enable_3rd_remote or use_mooncake_store_backend`。
- `update_default_config_from_user_config`（或 P800 等价入口）：把 `pp_rank/pp_size/total_layers` 从 `RankInfo` 注入 `cache_config`；**单节点（`nnodes==1`）时直接置 `node_layer_start=0, node_layer_end=num_layers`**（保证 Case 1 无后缀）。
- 校验调整：`use_mooncake_store_backend` 时放开"`enable_remote` 必须 `enable_ssd`"等约束（mooncake 无需 SSD staging）。
- **`enable_kv_sharing` 与 mooncake 互斥**：开启 mooncake 后 CFS/PCFS remote 被替代。

> ⚠️ **不确定点**：config.py 夹杂上游漂移（`cp_size`/`nnodes_per_pp_rank` 重构）。**只挑 mooncake 字段，不带入漂移重构**。P800 已有部分 rank 概念（命中 12），其与 mooncake 字段的对应关系须与我确认。

### 5.2 `flexkv/common/transfer.py`
- `TransferType` 新增 `REMOTE2H="REMOTE2H"`、`H2REMOTE="H2REMOTE"`。
- `TransferOp` 新增字段：`mooncake_store_block_hashes: Optional[np.ndarray] = None`（每元素对应 `src/dst_block_ids` 一个 block）。
- 新增 `_merge_remote2h_ops(ops, ...)`：合并 REMOTE2H/H2REMOTE 时保留 `mooncake_store_block_hashes`（`np.concatenate`）与 `src_block_node_ids`，避免被通用 `_merge_ops` 丢弃。
- batch merge `supported_types` 放开 `REMOTE2H/H2REMOTE`。
- 依赖编排（关键）：
  - **GET 非 layerwise**：attach `REMOTE2H`，令 `H2D` 依赖它（CPU buffer 填满后再 H2D）。
  - **GET layerwise**：`REMOTE2H` 作为 C++ layerwise pipeline 的**外部前置 stage**，独立 graph 节点，`LayerwiseTransferOp` 依赖它（保证 mooncake 前缀就绪后第一层才 fire eventfd，**C++ fused worker 不变**）。
  - **PUT**：merge `H2REMOTE`（专用 merger 保 hashes），令 `H2REMOTE` 依赖 `D2H`。
  - `batch_end_op_id` 优先级：GET `H2D > REMOTE2H > DISK2H`；PUT `H2REMOTE > H2DISK > D2H`。

### 5.3 `flexkv/transfer/worker_op.py`
- `WorkerTransferOp` 透传 `mooncake_store_block_hashes`。
- mooncake 后端（hashes 非空）时即便有 slot 也用 CPU `src/dst_block_ids` 计算指针。约 20 行。**保留 P800 已有 D2H profiling 字段差异**。

### 5.4 `flexkv/transfer/transfer_engine.py`
- 设备图谱新增 `H2REMOTE:(2,4)`、`REMOTE2H:(4,2)`（及对称 PEER 项，按 mooncake 版）。
- import `MooncakeStoreTransferWorker`、`PoolKind`。
- 主 KV 池：当 `use_mooncake_store_backend and _cpu_handle is not None` 且**无 `_remote_handle`** 时，创建 `MooncakeStoreTransferWorker.create_worker(... cpu_blocks=_cpu_handle.get_worker_tensor(), cache_config=..., pool_kind=PoolKind.KV)`，注册到 `_worker_map[H2REMOTE]`、`_worker_map[REMOTE2H]`（**同一 worker 双向复用**）。
- indexer sidecar 池：同理创建 `pool_kind=PoolKind.INDEXER, override_global_segment_size=0`（纯客户端，复用主池注册的 buffer，不重复占段），注册到 `_indexer_worker_map[H2REMOTE/REMOTE2H]`。
- **H2D/D2H 的 worker 注册保持 P800 原状（C1），仅新增 H2REMOTE/REMOTE2H 分支。**

### 5.5 `flexkv/cache/cache_engine.py`
- `self.use_mooncake_store_backend = cache_config.use_mooncake_store_backend`。
- `enable_remote` 时若 mooncake：`remote_cache_engine = MooncakeStoreCacheEngine(cache_config)`（取代 PCFS/Hierarchy 分支）。
- GET 组装 `op_remote2h`：当 mooncake 时，从 `sequence_meta.block_hashes` 切出 fragment3 对应区间传 `mooncake_store_block_hashes`（content-hash 寻址，不用文件 offset）。
- PUT 组装 `op_h2remote`：同理传 hashes，并 `add_dependency(op_h2remote, op_d2h)`。
- `shared_pcfs_read = enable_kv_sharing and index_accel and not use_mooncake_store_backend`（mooncake 不走 PCFS 共享读分支）。

### 5.6 `flexkv/storage/storage_engine.py`
- `enable_remote and use_mooncake_store_backend` 时**跳过 `RemoteAllocator`**：不分配独立 contributed/staging 段，hot CPU buffer 直接注册给 store（同步 PUSH 语义）。仅打印日志说明。

### 5.7 `flexkv/transfer_manager.py`（主要适配点，见 §6.2）
- 计算 `num_layers_on_node = sum(pp_rank_to_num_layers.values())` 与 `node_min_pp_start_layer`。
- mooncake 后端时回写：`node_layer_start=node_min_pp_start_layer`、`node_layer_end=node_min_pp_start_layer+num_layers_on_node`；`total_layers` 缺省时取 `model_config.num_layers`。供 `build_key` 生成 PP key 后缀。
- mooncake 后端 `remote_handle=None`（跳过 RemoteAllocator）。

### 5.8 构建与依赖
| 文件 | 改动 |
|---|---|
| `FlexKV/install.sh` | 新增源码构建 `mooncake-transfer-engine`（`--mooncake-version`）+ 安装 `mooncake-store` Python SDK |
| `FlexKV/setup.py` | **无需改动**（不引入 native 源文件，保持 P800 `csrc/*` 与 `-DCUDA_AVAILABLE`） |

### 5.9 sglang 侧
- **基本无需改动**，mooncake-store 对 sglang 透明。
- 仅保证 FlexKV 初始化能读到 `FLEXKV_USE_MOONCAKE_STORE_BACKEND` / `FLEXKV_MOONCAKE_STORE_CONFIG_PATH`（env 注入）。
- **不要**搬 mooncake 版 `flexkv_connector.py` 的 156 行差异（上游漂移 + P800 NUMA/profiling，与本特性无关）。

---

## 6. 关键流程时序

### 6.1 PUT（存储）时序
```
1. CacheEngine: match() 计算 fragment3（需写入 remote 的 block）
2. 组装 op_d2h (GPU→CPU, P800 CE) + op_h2remote (CPU→store, 带 block_hashes)
3. add_dependency(op_h2remote -> op_d2h)
4. TransferEngine 调度:
   a. GPUCPUTransferWorker 执行 D2H (CE 多路径)             [P800 资产]
   b. D2H 完成后 MooncakeStoreTransferWorker 执行 H2REMOTE:
      - _preprocess: 计算 cpu_ptr=base+blk_id*block_size, build_key
      - batch_put(keys, ptrs, sizes): 去重 → batch_put_from (RDMA 写)
```

### 6.2 GET（加载）时序
```
1. CacheEngine.match(): 先本地(GPU/CPU/SSD)，未命中走 remote_cache_engine
   = MooncakeStoreCacheEngine.match(): build_key 平铺 + batch_exists(_impl)
     → 最长(全 required 池)命中前缀 → MatchResultAccel(matched_pos="global")
2. 组装 op_remote2h (store→CPU, 带 block_hashes) + op_h2d (CPU→GPU, P800 CE)
3. 非 layerwise: add_dependency(op_h2d -> op_remote2h)
   layerwise:    REMOTE2H 作前置 stage, LayerwiseTransferOp 依赖之
4. TransferEngine 调度:
   a. MooncakeStoreTransferWorker 执行 REMOTE2H: batch_get_into (RDMA 读) 填 CPU
   b. CPU 就绪后 GPUCPUTransferWorker/layerwise 执行 H2D (CE 多路径)  [P800 资产]
```

---

## 7. 风险与应对（继承实施方案 §4，落到设计层）

| 编号 | 风险 | 设计层应对 |
|---|---|---|
| R1 | 上游版本漂移（最高优先级） | 禁整目录覆盖（C2）；逐文件按 hunk 提取 mooncake 专属改动；合并后过 lint+单测。**任何拿不准的 hunk 先问我。** |
| R2 | `transfer_manager.py` 的 rank/layer 模型差异 | **分两阶段**：①单机/单 PP 走 `build_key` Case 1 无后缀快路径，优先打通；②跨节点 PP 把 mooncake 的 `num_layers_on_node`/`node_min_pp_start_layer` 适配到 P800 rank 模型（基于现有 `cp_size`/`nnodes_per_pp_rank`）。映射关系须确认。 |
| R3 | mooncake-transfer-engine 在 P800 编译 + RDMA 打通（最大不确定性） | **M0 先做可行性预研**：确认 CPU 架构(x86/ARM)、RDMA 网卡(IB/RoCE)、verbs 驱动，能 build mooncake 及 SDK。参考 `nic.txt`。 |
| R4 | `cudaHostRegister` 兼容性 | P800 `transfer.cu` 已在用 `cudaHostRegister`（CUDA 兼容层），**沿用即可**。 |
| R5 | 与 C1 冲突（CE 被回退） | 合并 worker.py/transfer_engine.py/transfer.cu 前建立 §4 保护清单；**只接受 H2REMOTE/REMOTE2H 相关 hunk**，触及 `use_ce_transfer_*`/`GPUCPUTransferWorker`/`transfer.cu` CE 多路径的一律丢弃保留 P800 版。合并后跑 D2H/H2D 正确性回归。 |

---

## 8. 实施步骤与里程碑

| 阶段 | 内容 | 预估 |
|---|---|---|
| M0 预研 | mooncake-transfer-engine 在 P800 编译 + RDMA 连通性（R3） | 1–3 人天 |
| M1 新增 | `flexkv/external/` + worker.py 追加新类（§3/§4） | ~1 人天 |
| M2 接入 | config/transfer/transfer_engine/cache_engine/storage_engine（§5，守 C1） | 2–3 人天 |
| M3 单机打通 | 单机/单 PP 无后缀快路径，端到端 PUT/GET | 1–2 人天 |
| M4 跨节点 PP | `transfer_manager.py` layer-range 适配（R2），跨节点 key 隔离验证 | 2–3 人天 |
| M5 联调回归 | 多机 KV 共享 + P800 CE 回归（R5）+ 性能对比 | 2–3 人天 |

**总体：约 8–13 人天**，最大不确定性在 M0。

---

## 9. 配置、开关与安全（约束 C4）

### 9.1 环境变量
| 变量 | 作用 |
|---|---|
| `FLEXKV_USE_MOONCAKE_STORE_BACKEND=1` | 启用 mooncake-store 后端 |
| `FLEXKV_MOONCAKE_STORE_CONFIG_PATH=/path/mooncake_store.json` | 指定 store 配置文件 |
| `FLEXKV_DEBUG_MOONCAKE_STORE=1` | 打开 H2REMOTE/REMOTE2H 调试日志 |

### 9.2 配置文件 `mooncake_store.json`（字段见 §3.3.1）
```json
{
  "master_addr": "<ip:port>",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "device_name": "<rdma_dev>",
  "local_hostname": "<local_ip>",
  "global_segment_size": 256,
  "enable_ssd_offload": false,
  "ssd_offload_path": null,
  "master_metrics_port": 9003
}
```

### 9.3 安全要求
- **所有敏感参数（master 地址、RDMA 设备、hostname）只从配置文件/环境变量注入，禁止硬编码到代码。**
- **SSRF 防护**：`check_server` 会访问 `http://{master_ip}:{port}/get_all_segments`。该地址来自受信配置文件。**若需访问内网域名/内网 IP，须先与我确认**；默认拒绝对 `9.*/10.*/11.*/21.*/30.*` 等内网段的非预期访问。
- 启用 mooncake 后 CFS/PCFS remote 被替代，`enable_kv_sharing` 与之互斥。

---

## 10. 验证方案

1. **功能**：移植 `/mooncake/FlexKV/tests/test_mooncake_store_integration.py`；验证 `batch_put/batch_get/batch_exists` 与 `match()` prefix 命中。
2. **正确性**：跨实例写入后另一实例命中加载，逐 block 校验 KV 内容一致（可借 `FLEXKV_CHECKSUM_DEBUG`）。
3. **C1 回归**：开/关 mooncake 两模式下跑 P800 既有 D2H/H2D 正确性与性能用例，确认 CE 多路径行为与吞吐无回退。
4. **跨节点 PP**：PP=2/4 多拓扑验证 key 后缀隔离（不同拓扑不串 cache）。
5. **性能**：对比 CFS/PCFS，记录 H2REMOTE/REMOTE2H 带宽与端到端 TTFT 收益。

---

## 11. 附录：关键代码位置索引

| 内容 | 参考实现位置（`/mooncake/FlexKV`） |
|---|---|
| Key 规则 | `flexkv/external/mooncake_store_keys.py` |
| Config/Client/CacheEngine | `flexkv/external/mooncake_store_utils.py` |
| 传输 worker | `flexkv/transfer/worker.py::MooncakeStoreTransferWorker`（L2157+） |
| 类型/合并 | `flexkv/common/transfer.py`（`TransferType`、`_merge_remote2h_ops`） |
| worker_op 透传 | `flexkv/transfer/worker_op.py`（L19-40） |
| 引擎注册 | `flexkv/transfer/transfer_engine.py`（L327-347、L618-638） |
| cache 接入 | `flexkv/cache/cache_engine.py`（L450-462、L774-796、L1296-1313） |
| storage 跳过 allocator | `flexkv/storage/storage_engine.py`（L113-131） |
| rank/layer 回写 | `flexkv/transfer_manager.py`（L149-195） |
| config 字段/校验 | `flexkv/common/config.py`（L347-396、L620-686） |

> **再次提醒：本文档中任何标注「不确定点 / 须确认」之处，以及实际编码中遇到的拿不准的合并取舍，请务必先与我确认，不要自作主张。**
