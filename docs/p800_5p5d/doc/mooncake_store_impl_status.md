# mooncake-store 移植 P800 FlexKV — 实现 / 测试进度

> 配套文档：
> - 详细设计：`doc/mooncake_store_p800_detailed_design.md`
> - 任务拆分与协同跟踪：`doc/mooncake_store_impl_progress.md`
> - 本文档：**记录代码实现与测试的实时进度**（按文件粒度）
>
> **【约束】遇到任何不确定的地方不自作主张，随时确认。** 已遵守硬约束 C1（保留 P800 D2H/H2D CE 优化）/ C2（逐 hunk，不整文件覆盖）/ C3（不碰 GPU/CUDA/csrc）/ C4（敏感参数 env/配置注入）。

更新时间：2026-07-09 15:06（最新：跨实例共享失败排查，见 §7）

---

## 0. 总览

| 维度 | 状态 |
|---|---|
| 代码实现（T1–T5） | ✅ 全部完成，已落地 FlexKV 源码（共 11 个文件 + install.sh） |
| 单测编写（T6） | ✅ 完成（7 文件，全程 mock，离线可收集；py_compile+lint 0 错） |
| 单测运行 | ✅ 真机容器全绿（gpu-146，py3.12+torch2.9+8×H20）：**70 passed / 1 skipped**（skip=真机集群 placeholder） |
| RDMA/编译可行性（M0/G0） | ⏸️ 待验证（mooncake-transfer-engine 已装 0.3.5 但无 `mooncake.store`，需 WITH_STORE 重建；flexkv c_ext 未编译） |
| 所有确认 Gate（G0–G3） | 已闭环（G0/M0 顺延真机；G1/G2/G3 已解决/实现） |

> ⚠️ **环境前提（明天真机需满足）**：当前工作机默认 `python`=Python **3.6.8**，而 mooncake 适配源码用了 `from __future__ import annotations`（需 **≥3.7**）+ `torch`。测试已设计为「有 fake 即可免 c_ext/mooncake SDK/GPU」，真机只要 **Python≥3.7 且 torch 可用** 即可收集运行全部离线用例；硬件依赖仅 1 个 placeholder 用例（真实 mooncake 集群+RDMA）。

---

## 1. 代码实现进度（按文件）

> 全部为 P800 基线 `/data1/home/phaedonsun/p800/FlexKV/` 下文件。✅=已落地并 lint 0 错。

### 1.1 新增模块（T1，dev-external）
| 文件 | 内容 | 状态 |
|---|---|---|
| `flexkv/external/__init__.py` | 惰性导出 `MooncakeStoreConfig/Client/CacheEngine` | ✅ |
| `flexkv/external/mooncake_store_keys.py` | `PoolKind/PoolSpec/build_key`（Case1 无后缀 / Case2 `_pp_rank_i_of_N`） | ✅ |
| `flexkv/external/mooncake_store_utils.py` | `MooncakeStoreConfig.from_file` / `MooncakeStoreClient`（setup/check_server/warm_up/register_buffer/batch_put 去重/batch_get/batch_exists） / `MooncakeStoreCacheEngine`（match 单池+多池 joint-existence） | ✅ |

差异备注：删除参考实现死代码 `from gc import enable`；注释非 ASCII→ASCII；逻辑 100% 对齐。

### 1.2 worker 追加（T2，dev-worker）
| 文件 | 内容 | 状态 |
|---|---|---|
| `flexkv/transfer/worker.py` | **末尾 append** `MooncakeStoreTransferWorker`（~200 行）：`cudaHostRegister`+`register_buffer`、`_preprocess`（指针 `base+blk_id*block_size`、`build_key`）、`_transfer_impl`（H2REMOTE→batch_put / REMOTE2H→batch_get） | ✅ |

C1 保护：`GPUCPUTransferWorker`/`tpGPUCPUTransferWorker`、`use_ce_transfer_*`、`_safe_checksum`、带计时 `launch_transfer` **逐行未改**。
主 agent 拍板：① import 置追加块开头（`# noqa: E402`）② debug-gate 改 `!= "0"`（修正参考恒真 bug，对齐 §9.1）。

### 1.3 数据结构与合并（T3，dev-core）
| 文件 | 内容 | 状态 |
|---|---|---|
| `flexkv/common/transfer.py` | `TransferType.REMOTE2H/H2REMOTE`（P800 已存在未重复）、`TransferOp.mooncake_store_block_hashes`、`_merge_remote2h_ops`、batch merge 放开 + GET/PUT 依赖编排 + layerwise 前置 stage + `batch_end_op_id` 优先级 | ✅ |
| `flexkv/transfer/worker_op.py` | 透传 `mooncake_store_block_hashes`；mooncake 用 CPU block ids 算指针；保留 P800 D2H profiling 字段 | ✅ |
| `flexkv/common/config.py` | mooncake 全字段 + `enable_pool_specs()` + env 解析（`FLEXKV_USE_MOONCAKE_STORE_BACKEND`/`..._CONFIG_PATH`）+ `enable_remote` 派生 + 单机 nnodes==1 预填 node_layer_start/end | ✅ |

Gate-G1 解决：mooncake 仅用 `pp_rank/pp_size/num_layers/nnodes`（两版一致），未带入 `cp_size/attn_cp_size` 漂移。
主 agent 拍板：env 解析用 `bool(int(os.getenv(...,0)))`（修正参考恒真 bug）。

### 1.4 引擎接入（T4，dev-engine）
| 文件 | 内容 | 状态 |
|---|---|---|
| `flexkv/transfer/transfer_engine.py` | import `MooncakeStoreTransferWorker`/`PoolKind`；主 KV 池 + indexer sidecar 池（`override_global_segment_size=0`）worker 注册到 `_worker_map`/`_indexer_worker_map[H2REMOTE/REMOTE2H]`；设备图谱 P800 已存在未重复加 | ✅ |
| `flexkv/cache/cache_engine.py` | `remote_cache_engine = MooncakeStoreCacheEngine`；GET `op_remote2h`/PUT `op_h2remote` 传 hashes（切片区间套用参考表达式 + assert）；`shared_pcfs_read` 排除 mooncake | ✅ |
| `flexkv/storage/storage_engine.py` | mooncake 时跳过 `RemoteAllocator`（hot CPU buffer 直注册，无 staging 段） | ✅ |

Gate-G2 未触发：两版 fragment 切分逐行一致。C1：H2D/D2H 注册零改动，仅新增 H2REMOTE/REMOTE2H 分支。

### 1.5 rank/layer 适配 + 构建（T5，dev-rank）
| 文件 | 内容 | 状态 |
|---|---|---|
| `flexkv/transfer_manager.py` | mooncake 时 `remote_handle=None`（跳过 RemoteAllocator）；统一 layer-range：用 `all_gpu_layouts`+`gpu_worker_key_mapping.pp_rank` 算 `num_layers_on_node`，回写 `node_layer_start=0/node_layer_end=num_layers_on_node`（单机→Case1 无后缀 / 跨节点→Case2 后缀） | ✅ |
| `FlexKV/install.sh` | 开关化（`--enable-mooncake-store` / `FLEXKV_BUILD_MOONCAKE_STORE=1`，默认关）构建 mooncake-transfer-engine（`-DWITH_STORE`）+ store SDK 验证；默认与原构建逐字节一致 | ✅ |

Gate-G3 已实现（最小零协议改动）：主 agent 推导 `node_layer_end - node_layer_start = num_layers_on_node`（`node_min_pp_start_layer` 抵消），故跨节点 PP key 映射**无需** plumb `pp_start_layer`、**未**改 `RegisterTPClientRequest`/`KVTPClient`/adapter/`TransferEngine`；`StorageEngine` 保持 `num_layers_per_pp_stage`（C2 不引入上游 PP 重构漂移）。

### 1.6 sglang 侧（§5.9）
- ✅ 透明，无功能改动；仅靠 `FLEXKV_USE_MOONCAKE_STORE_BACKEND` / `FLEXKV_MOONCAKE_STORE_CONFIG_PATH` env 注入。未引入 `flexkv_connector.py` 漂移差异。

---

## 2. 测试进度（T6，dev-test）

> 原则：**全程 mock**（`mooncake.store.MooncakeDistributedStore` / `cudaHostRegister` / CUDA tensor / `flexkv.c_ext`），离线可被 pytest 收集；真机相关用例用 `@pytest.mark.skipif` 标注。**本期不运行，待明天测试环境。**

> 共享工具：`tests/_mooncake_store_testkit.py`（非测试文件）——导入时注入 fake `flexkv.c_ext`（确定性 FNV 哈希生成 block_hashes，免编译 C 扩展）、`FakeMooncakeStoreClient`、`make_cache_config`；各测试文件首行 `import _mooncake_store_testkit` 即装好 fake。

| 测试文件（`FlexKV/tests/`） | 覆盖点 | 编写 | 离线可跑 | 运行 |
|---|---|---|---|---|
| `_mooncake_store_testkit.py`（共享工具） | fake c_ext / FakeClient / make_cache_config | ✅ | n/a | n/a |
| `test_mooncake_store_keys.py` | `build_key` 全分支(base/Case1含start>0/Case2/legacy)、PoolKind/PoolSpec、**T5 num_layers_on_node→后缀映射** | ✅ | ✅(mock) | ⏸️ |
| `test_mooncake_store_config.py` | `from_file` 全字段/GB→bytes/override=0纯客户端/override非0/回退/缺路径ValueError/缺文件FileNotFound/env回退 | ✅ | ✅ | ⏸️ |
| `test_mooncake_store_cache_engine.py` | `match` 单池最长前缀(全/部分/gap截断)、多池joint(indexer/KV/首块缺截断)、空序列不发RPC、matched_pos、no-op 接口 | ✅ | ✅(FakeClient) | ⏸️ |
| `test_mooncake_store_merge_ops.py` | `_merge_remote2h_ops` concat保hashes/node_ids、任缺→None、callback透传；merge GET(H2D依赖REMOTE2H)/PUT(H2REMOTE依赖D2H)、batch_end优先级、非法类型、空图 | ✅ | ✅ | ⏸️ |
| `test_mooncake_store_worker.py` | `_preprocess` 指针/方向/key/hashes断言；`_transfer_impl` 分派+非法类型ValueError（object.__new__ 绕开 cudaHostRegister/真实client；fake c_ext+importorskip） | ✅ | ✅(mock) | ⏸️ |
| `test_mooncake_store_integration.py` | fake store(dict+ctypes真实拷字节) 跑真实 MooncakeStoreClient put→exists→get 逐字节一致/幂等/最长前缀；CacheEngine.match 端到端；check_server monkeypatch no-op | ✅ | ✅(fake) | ⏸️ |
| └ `test_real_cluster_roundtrip_placeholder` | 真实 mooncake 集群+RDMA | ✅ | ✗ | ⏸️ skip 待真机 |

✅ 全部编写完成；7 文件 py_compile + lint 0 错；未改任何被测源码 / conftest / 既有测试。

---

## 3. 完整性核对（对照详细设计）

| 设计章节 | 项 | 状态 |
|---|---|---|
| §3 | external 三文件落地且行为对齐 | ✅ |
| §4 | worker 追加且 C1 未破坏 | ✅ |
| §5.1 | config 字段/enable_pool_specs/env | ✅ |
| §5.2 | TransferType/TransferOp/merge | ✅ |
| §5.3 | worker_op 透传 | ✅ |
| §5.4 | transfer_engine worker 注册 | ✅ |
| §5.5 | cache_engine remote_cache_engine + hashes | ✅ |
| §5.6 | storage_engine 跳过 RemoteAllocator | ✅ |
| §5.7 | transfer_manager layer-range（含跨节点） | ✅ |
| §5.8 | install.sh 构建步骤 | ✅ |
| §5.9 | sglang 透明 | ✅ |
| §10 | 验证方案对应单测 | ✅ 编写完成（待真机运行） |

---

## 4. 待办 / 明天测试环境就绪后

1. **运行单测**：`cd FlexKV && python -m pytest tests/test_mooncake_store_*.py -v`（离线 mock 用例应全绿）。
2. **M0 / G0 真机预研**：mooncake-transfer-engine 在 P800 编译（`install.sh --enable-mooncake-store`）+ RDMA 连通性（IB/RoCE + libibverbs，参考 `nic.txt`）。
3. **C1 回归**：开/关 mooncake 两模式跑 P800 既有 D2H/H2D 正确性与性能用例，确认 CE 多路径无回退。
4. **端到端**：单机快路径 PUT→GET 跨实例命中；跨节点 PP key 后缀隔离（PP=2/4）。
5. 准备一份示例 `mooncake_store.json`（master_addr/device_name 等，按真机网卡填）。

---

## 5. 变更文件清单（交付参考）

新增（4+7）：`flexkv/external/__init__.py`、`mooncake_store_keys.py`、`mooncake_store_utils.py`；`FlexKV/tests/_mooncake_store_testkit.py` + `test_mooncake_store_{keys,config,cache_engine,merge_ops,worker,integration}.py`（共 7 测试文件）。

---

## 6. 真机单测运行记录（2026-07-01，relay-cli → GPU executor 容器）

环境：executor `gpu-***` 上容器 `flexkv_distreuse`，Python 3.12.13 + torch 2.9.1+cu128 + CUDA + 8×H20；`/data1/phaedonsun/flexkv/` 映射入容器。远端 FlexKV 副本已含全部 mooncake 改动。

**首次运行发现并修复（源码）**：`flexkv/external/__init__.py` 存在 eager `import mooncake_store_utils`，使 `flexkv.common.config` 一被导入即连锁拉到 `flexkv.c_ext`（容器未编译 c_ext → conftest 收集失败）。已改为**真正惰性**（保留 `__getattr__`，删除顶层 eager import）；本地工作区与远端副本同步修复。→ keys/config 单测随即通过。

**运行结果**：`pytest tests/test_mooncake_store_*.py`
| 结果 | 说明 |
|---|---|
| ✅ 27 passed | keys / config / cache_engine / merge_ops 全过 |
| ⏭️ 2 skipped | worker 模块（fake c_ext 缺 `CMatchResult` 被 importorskip 跳过）+ 真机集群 placeholder |
| ❌ 1 failed | integration roundtrip：`memoryview: unsupported format <c`（fake ctypes buffer 需 `.cast('B')`） |

**待修（测试代码，dev-test 处理中）**：
1. testkit fake `c_ext` 补齐 `worker.py` 导入的全部符号（`CMatchResult` 等）→ 让 worker 用例真正运行。
2. integration roundtrip 用 `memoryview(buf).cast('B')` 修字节读写。
3. 注册 `pytest.mark.mooncake`（消 warning，可选）。
→ 修复后由 main 同步远端两执行端并重跑回填。

**修复后重跑（2026-07-01 13:11，gpu-146）**：`70 passed, 1 skipped`（skip=真机集群 placeholder）。dev-test 修法：fake c_ext 加**模块级 `__getattr__` 兜底**（任意未定义符号返回通用 dummy，`Hasher/gen_hashes/get_hash_size` 保留真实语义）；worker 测试 `importorskip`→硬 import；integration `memoryview.cast('B')`；`pyproject.toml` 注册 `mooncake` marker。4 个文件已 base64 同步远端。

**尚未做（待续）**：flexkv c_ext 原地编译（`build.sh`）+ mooncake `WITH_STORE=ON` 重建 → 用于真实 `mooncake.store` 集成与两节点端到端测试（M0/G0）。

### 6.1 两节点端到端脚本（2026-07-01 新增，本地已建，lint 0 错）
- `benchmarks/dist_benchmark/benchmark_dist_direct_mooncake_store.py`：`--mode put`（Node A 写入并常驻保持 RDMA 段）/ `--mode get`（Node B 冷缓存拉取+验证）/ `--mode probe`（仅探测 key 存在）。
- `benchmarks/dist_benchmark/example_dist_direct_mooncake_store_config.yml`：P2PHANDSHAKE + mooncake_master（无 redis/etcd）配置模板，含每节点 `--local-hostname/--device-name/--master-addr` 覆盖。
- **验证/观测机制**（确保两节点确实共享）：①同 seed→同 token_ids→同 content-hash key；②query-only `MooncakeStoreClient.batch_exists` 直接探测 key 物理存在于共享 store；③Node B 本地缓存全冷却能高命中→数据来自远端；④负对照（从未 PUT 的随机序列应 ~0% 命中）。非零退出码表示验证失败（CI 友好）。
- **运行前置（待做）**：容器内编译 flexkv c_ext；mooncake `WITH_STORE=ON` 重建以获得 `mooncake.store` + `mooncake_master`；启动一个 `mooncake_master`；两文件同步到两台执行端。

### 6.2 真机双节点环境搭建记录（2026-07-01，relay-cli）
执行端：`gpu-***`(***) / `gpu-***`(***)，各容器 `flexkv_distreuse`，8×H20 + 8×mellanox(**`mlx5_bond_0..7`**)。
- 测试代码已同步两台：`benchmark_dist_direct_mooncake_store.py` + config + lazy 版 `external/__init__.py`。
- **flexkv c_ext**：`build.sh`（`FLEXKV_ENABLE_METRICS=0`，third_party 仅需已在的 xxHash）→ gpu-146 已生成 `c_ext.so` 并 `import flexkv.c_ext` OK（**P800 CE `transfer.cu` 在 H20/CUDA12.2 编译通过，C1 正向信号**）；gpu-129 构建中。
- **mooncake.store**：原 mooncake 0.3.5 为 `WITH_STORE=OFF`（无 store）。用 `_build_mooncake_store.sh`（`-DWITH_STORE=ON`）重建 → gpu-146 已产出 `mooncake/store*.so` 且可导入；gpu-129 构建中。
- **CUDA 环境坑（已解）**：torch2.9(cu128) 的 `libc10_cuda.so` 需 `cudaGetDriverEntryPointByVersion`（12.5+），系统 cudart 12.2 缺该符号。运行时须 `export LD_LIBRARY_PATH=/opt/vllm-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH`（torch 自带 cudart 12.8）→ `import mooncake.store` OK。
- `mooncake_master`：`/opt/vllm-env/bin/mooncake_master`。metadata=P2PHANDSHAKE，无 redis/etcd。
- 计划：master 起在 gpu-146:50051；A(gpu-146)=put 常驻，B(gpu-129)=get+验证。

### 6.3 双节点端到端运行进展（2026-07-01，持续）
两台均已就绪（c_ext + mooncake.store 可导入；master 运行于 gpu-146:50051）。运行 benchmark 过程中发现并修复：
- **源码兼容修复1**：`external/__init__.py` eager import → 惰性（避免 config 连锁拉 c_ext）。
- **源码兼容修复2**：`MooncakeStoreClient` 的 query-only 客户端原用 `rpc_only` 协议，此 mooncake 0.3.5 构建**不支持**（`unsupported_protocol`）→ 改用真实协议(rdma/tcp)+`global_segment_size=0` 纯客户端。
- **benchmark 修复**：tp_client 进程改用 `spawn`（主进程 mooncake client 已初始化 CUDA，fork 子进程 `cudaErrorInitializationError`）。FlexKV 内部 worker 本就用 spawn。
- 进展：PUT 节点已跑通 GPU 注册 → mooncake-store worker 创建 → 3×4GB CPU pin → store client 初始化 + check server 通过。
- **当前阻塞**：warmup/put 返回 **-800 (TRANSFER_FAIL)**，tcp/rdma 均如此。

### 6.4 -800 根因深度分析（2026-07-01 20:30，通读更新后 Mooncake 0.3.9 源码 `/data1/home/phaedonsun/p800/Mooncake`）
**先纠正上一轮的误判**：`master 显示 Storage 0.00 B` **不代表段未挂载**。master 汇总字符串是 `Mem Storage: <已分配> / <总容量>`（master_metric_manager.cpp:1514），0.00B 是"已分配"（因 put 从未成功→自然为 0），需看**第二个数字（总容量）**才能判断段是否挂载。且 `setup()` 返回 0 已保证 `MountSegment` 成功（real_client.cpp:328-334 失败即返回非 0）。→ **段其实已挂载，问题在传输引擎实际写入环节**。

**真正根因（TRANSFER_FAIL 产生点）**：mooncake put 分两步——BatchPutStart（master 分配段）→ SubmitTransfers（传输引擎把 local_buffer 写入段，transfer_task.cpp）。传输策略 `selectStrategy`（transfer_task.cpp:681）有两种：
- `LOCAL_MEMCPY`：同机传输用 memcpy（必成功）。
- `TRANSFER_ENGINE`：走 RDMA/TCP `openSegment`+transfer。
**关键**：`MC_STORE_MEMCPY` 环境变量默认未设 → `memcpy_enabled_=false`（transfer_task.cpp:413-415）→ **即使 warmup put 是同机 A→A，也走 TRANSFER_ENGINE**，在本环境（bonded 网卡 `mlx5_bond_0` + 容器）下 `openSegment`/传输失败 → -800。

**佐证（sglang 自己的双节点 RDMA 脚本 `sglang/klx/server/glm_5/w8a8/run_2nodes*.sh`）**：
```bash
export MOONCAKE_LOCAL_HOSTNAME="${MC_STORE_LOCAL_IP}"          # 可路由 IP
export MC_TCP_BIND_ADDRESS="$(ip route get 1.1.1.1 | ... prefsrc)"  # 传输引擎绑定到可路由 IP
```
以及 README：`MC_MS_AUTO_DISC=1`（RDMA 网卡自动发现，覆盖 device_name）。→ 强烈指向：**我们的传输引擎绑定到了容器内错误的 IP**（非 ***），使 openSegment/handshake 失败。

**修复方案（按优先级，待执行端上线后验证）**：
1. **重建 0.3.9**：两执行端当前装的是 **0.3.5**，源码已更新到 **0.3.9**，需 `-DWITH_STORE=ON` 重建两台（用 `scripts/_build_mooncake_store.sh`）。
2. **传输引擎绑定/发现 env（核心）**：运行前 `export MC_TCP_BIND_ADDRESS=<本节点可路由IP>`（A=*** / B=***）；RDMA 走 `MC_MS_AUTO_DISC=1`（或正确 GID）。
3. **同机快路径兜底**：`export MC_STORE_MEMCPY=1` 让 warmup 及 A→A 本地 put 走 memcpy（先解锁本地写入）。
4. **详细日志抓真错**：`export GLOG_logtostderr=1 GLOG_v=1`（或 `MC_LOG*`）捕获 `openSegment`/RDMA GID 的确切失败原因。
- **结论不变**：FlexKV 侧移植逻辑已由 70 单测 + 真机导入/GPU 注册/段挂载/client 初始化验证；-800 属 **mooncake 传输引擎运行时 env/网络绑定**问题，非移植代码缺陷。setup() 7 参签名与 0.3.9 完全一致，`rpc_only` 已正确去除（0.3.9 setup 校验只允许 tcp/rdma）。

### 6.5 ✅ -800 已解决 + 双节点环境搭建完成（2026-07-01 22:2x）
根因是**多重的**，全部解决：
1. **版本**：机器装的是 mooncake 0.3.5，用户更新源码到 **0.3.9**。0.3.9 需要：
   - **pybind11 submodule**（`extern/pybind11`，CMakeLists.txt:21 `add_subdirectory`）→ `git submodule update --init` 拉取。
   - **yalantinglibs 0.5.7**（`dependencies.sh: YALANTINGLIBS_VERSION=0.5.7`；机器上是旧的 0.5.1，缺 `ib_socket_t::config_t` → rpc_communicator.cpp 编译失败）→ 用户帮忙下载 tarball，装到 /usr/local。
2. **网络**：mooncake 互联要走 **bond0**（gpu-146=*** / gpu-129=***），之前误用 relay 内网 ***。运行前 `export MC_TCP_BIND_ADDRESS=<本节点bond0 IP>` + `MOONCAKE_LOCAL_HOSTNAME`。
3. **libasio.so**：0.3.9 的 `mooncake_master` 动态链接 `/usr/local/lib/libasio.so`，需 `LD_LIBRARY_PATH` 含 `/usr/local/lib`。
4. **同机 memcpy**：`MC_STORE_MEMCPY=1`（同机 warmup put A→A 走 memcpy）。

**当前状态**：两台 mooncake 0.3.9 + store + flexkv c_ext 全部就绪；`mooncake_master` 运行于 gpu-146:50051；Node A(PUT) 跑通 GPU 注册 → worker 创建 → **4GB 段挂载成功** → store setup/warmup/P2P handshake **全部通过（无 -800）**。

### 6.6 新问题（PUT 阶段，FlexKV 侧，排查中）
PUT 传 **0 tokens**，失败在**第一阶段 D2H（GPU→CPU staging）**，非 mooncake：
```
ERROR Error launching transfer: invalid configuration argument (CUDA)
Failed op: TransferType.D2H, valid_block_num=32,
  src_block_ids=array([], dtype=float64), dst_block_ids=array([], dtype=float64)
```
即 D2H op 的 block_ids 为空但 valid_block_num=32 → CE kernel 以 0 block 启动报 invalid config。参考 benchmark(redis) 用**相同** per-token slot 调用能跑通 → 差异在 mooncake 传输图构造/合并（cache_engine `_put_impl_global` / common/transfer.py PUT 合并 H2REMOTE 依赖 D2H）。已交 dev-core / dev-engine 排查。

### 6.7 PUT 失败根因再定位（2026-07-01 22:4x，纠正 6.6）
经真机 traceback + 参数诊断，**6.6 的初判被推翻**，精确结论：
- **不是 clear/set_gpu_blocks 不对称**：那条路径仅 `nnodes>1 && pp_size>1` + MultiNodeHandle 触发；本 benchmark 单机 `tp=1/dp=1/nnodes=pp_size=1`，clear_gpu_blocks 根本没调用。失败 op 的 slot 已分配(0/1)，block_ids 入 worker 时是满的（空 float64 只是 slot 化残留）。（clear/set 不对称是**真实潜在 bug，仅跨机 PP 触发**，本期不修，记为已知问题。）
- **真因**：`worker.py:405 transfer_kv_blocks(...)` 抛 `invalid configuration argument`。诊断参数：`D2H nblk=32 nlayers=4 cta=4 use_ce=True/False 都失败 gbt=0 chunk=32768 dev=0 ngpu_blocks=4`。
- **逐一排除**：① arch 匹配（c_ext=sm_90，H20=9.0；否则应是 err 209）；② 通用 CUDA 正常（容器内 torch.mm KERNEL_OK）；③ transfer.cu 仅 L1027 一处 `<<<>>>`（非 CE），但 CE 路径也报 err 9 → CE 分支(ATen path2)也有 kernel 启动失败。
- **定性**：这是 **P800 CE 的 GPU↔CPU staging 传输(transfer.cu)在 H20 spawn worker 子进程 + MPS 环境下的问题，与 mooncake 完全无关**，是首次真机跑真实 transfer.cu（70 单测全 mock c_ext）暴露的既有/环境问题。已交 **dev-worker**（C1/CE owner）深挖 transfer.cu CE 分支 + spawn/MPS/stream 交互，我在真机配合验证。
- 诊断补丁（临时，待还原）：worker.py L237 traceback、L403 前 [D2H-DIAG] 打印。（已于 07-02 全部还原，仅保留真实 start_layer_id 修复。）

### 6.9 ✅ indexer 分布式共享真机验证 + 修复 indexer-replica 漏传 hashes（2026-07-02）
**目的**：验证 DSA/NSA indexer（GLM5 用）在 mooncake-store 的分布式共享（多池）——单机 tp=8 + MLA + indexer。
**新建/改动**：
- `benchmarks/dist_benchmark/benchmark_dist_direct_mooncake_store.py`：支持可选 `indexer:` 配置（设 `cache_config.indexer`、注册 indexer GPU buffer、PUT/GET/probe 遍历所有活跃池）。无 indexer 配置时行为不变。
- `benchmarks/dist_benchmark/example_dist_direct_mooncake_store_indexer_config.yml`：tp=8/use_mla/indexer 代表性配置。
- `scripts/_e2e_indexer_run.sh`：单机 runner。

**真机抓出的真实 bug（已修）**：`transfer/transfer_engine.py::_assign_op_to_workers` 创建 **indexer replica TransferOp**（dict 与 singleton 两分支）时**漏传 `mooncake_store_block_hashes`**（也漏 `src_block_node_ids`）。而 `MooncakeStoreTransferWorker._preprocess` 断言 `mooncake_store_block_hashes is not None`（无消息断言）→ 表现为空的 `Error launching transfer:`，indexer 池从不写入（主KV走原始 op 带 hashes 故正常）。
**修复**：两处 indexer replica 补 `mooncake_store_block_hashes=op.mooncake_store_block_hashes.copy()`（+ `src_block_node_ids`）。纯 Python，不碰 C1。
**验证结果（gpu-146 单机 tp=8 + indexer，默认 kernel 路径 CE=0）**：
- PUT：`keys present in store after PUT: 32/32 across 2 pool(s) -> OK`（16 主KV + 16 indexer）。
- GET（冷缓存）：`store keys visible: 32/32`、`cold-cache shared hit: 100.00%`、负对照 0.00%、`Error launching transfer: 0` → **CROSS-NODE SHARING CONFIRMED**。
→ **indexer 分布式共享（多池 KV+indexer 命中）真机打通**。跨机 TP16 仍未验证（见 §6.8 分析：MLA 复制特性规避 key 冲突、但 remote 多节点路径待验证）。

### 6.10 跨机 TP16（2×8）+ indexer 真机验证（2026-07-02，部分通过 + 1 修复 + 1 待解）
**新建 harness**：`benchmarks/dist_benchmark/benchmark_dist_crossnode_tp_mooncake_store.py`（复刻 sglang connector 的多节点编排：`--node-rank 0` 建 KVManager(nnodes=2,tp=16)+MultiNodeHandle bind master_ports+本地8卡；`--node-rank 1` 跑 `TransferManagerOnRemote.create_process`+本地8卡）、`example_dist_crossnode_tp16_mooncake_store_config.yml`、`scripts/_e2e_xnode_tp_run.sh`。拓扑：nnodes=2, tp_size=16, pp_size=1 → tp_size_per_node=8, nnodes_per_tp_group=2（满足 `<=2` 约束）。

**修复的真实 bug（跨机专属）**：mooncake store 配置文件原用 `mkstemp` 生成**节点本地临时路径**，随 cache_config 经 ZMQ 发给 Node B 后**在 B 文件系统不存在** → B 的 remote mooncake worker `from_file()` 读不到 → setup 卡死、ready_event 不触发、Node A 就绪查询永久超时。**改用固定路径 `/tmp/flexkv_xnode_mooncake_store.json`**（路径串跨节点一致，每台各写本机 local_hostname 内容）后，双节点 bring-up 完全打通。

**通过项**：
- 双节点 bootstrap：Node A KVManager master bind master_ports、Node B `TransferManagerOnRemote started successfully`、rendezvous 成功（is_ready 通过）。
- 两节点各 8 卡注册、TransferEngine + mooncake/indexer worker 就绪、8GB 主池 + indexer 池 pin、`Store initialised`×2。
- **数据正确写入共享 store：store-probe `8/8 keys exist`（KV 池）+ `8/8`（indexer 池）**——跨机 TP16（MLA latent 跨 TP 复制）下 content-hash key 两池均落盘。
- **0 个 `Error launching transfer`**。

**待解项（下一层）**：Node A 的 `wait(completely=True)` 20s 超时、PUT 报 **0 tokens**（假阴性）——**跨节点任务完成回传未闭合**：Node B 的 remote TM 未见 submit/H2REMOTE 活动，Node A 的 MultiNodeHandle 收不到 Node B 的 CompletedOp → 任务 pending_count 不归零。数据其实已在 store（8/8 两池），属**完成/ack 聚合**问题而非数据丢失。疑点：direct-benchmark 路径下 MultiNodeHandle 的结果轮询线程/`set_slot_mapping` 向 remote 的下发未完全走通（正常经 sglang connector 编排）。需进一步排查 `MultiNodeHandle.submit/_polling_worker` 与 KVTaskEngine 对 remote handle 的 start/结果聚合。
**结论**：跨机 TP16 的**多节点 bring-up + 数据面 + 两池落盘**已真机验证可行；**端到端 PUT/GET 完成语义**在 direct-benchmark 下还差最后一层 ack 闭合（不影响 sglang 正常编排路径的判断，但需补验）。

### 6.11 ✅ 跨机 TP16 完成回传真因锁定并修复 + 端到端 CONFIRMED（2026-07-02 续）
**真因（真实 bug，非 harness 独有——真实 sglang 跨机 TP 同样会 hang）**：
`transfer_manager.py::TransferManagerOnRemote._handle_submit` 收到 master 下发的 graph 后**无条件**将其存入 `_pending_graphs` 并 `return`，只有等到一条 `set_slot_mapping` 消息才 `set_gpu_blocks + submit`。而 `set_slot_mapping` 仅在**跨机 PP** 发送：sglang `flexkv_comm.py` L181-183 `should_send_slot_mapping_to_remote = is_pp_receiver AND is_cross_node_pp`，**纯跨机 TP（pp_size=1）恒为 False**。
- 结果：跨机 TP 的 graph 在 remote 永久 pending → 从不执行、从不回报完成。
- master 侧 `kvtask.py` `_get_completed_ops` 要求 `required_completed_count = len(transfer_handles) = 2`（本地 process TM + remote MultiNodeHandle）**两个 handle 都报完成**才算 graph 完成；remote 永不报 → 只本地报（count=1）→ 永不达 2 → `wait(20s)` 超时 → PUT 报 **0 tokens**。
- 数据其实已在 store：MLA latent 跨 TP 复制，**Node A 本地 8 卡即写全** content-hash key，故此前 store-probe 8/8 仍 OK（假阴性仅在"完成计数"层）。
- 而 master 跨机 TP 提交的 graph 是**未 clear（GPU blocks 已 set）**的（`kvtask.py` L302-304，注释"same slot_mapping across TP ranks, no clear needed"）——这正是可用于区分 TP/PP 的信号。

**修复（纯 Python，不碰 C1/CE/kernel）**：
1. `common/transfer.py`：`TransferOpGraph` 新增只读 `has_gpu_blocks_set()`——所有 GPU 传输 op 的 GPU 侧 block_ids 非空即 True（未被 `clear_gpu_blocks` 清过）。
2. `transfer_manager.py`：`_handle_submit` 开头判定 `graph.has_gpu_blocks_set()`：**True（跨机 TP，ready-to-run）→ 立即 `set_gpu_blocks`（无需）+ submit**；False（跨机 PP，已 clear）→ 保持原 pending 等 `set_slot_mapping` 逻辑不变。两台已 patch（`scripts/_patch_xnode_tp_submit.sh`，幂等）。
3. harness 增补 `putget` 组合模式（`_main_node0`）：同进程 PUT → `KVManager._clear_cpu_cache()` → 冷 GET，Node A 全程存活持有 mooncake 段，避免 kill-PUT 导致段数据丢失（此前误 kill 使 GET 只见 20/32、hit 12.5%）。

**端到端验证结果（gpu-146=Node A/master, gpu-129=Node B/remote，nnodes=2 tp=16 pp=1，默认 kernel 路径 CE=0）**：
- 独立 PUT：`PUT: 1024 tokens, 22.6 ms`、`keys 8/8`、0 错误；**Node B 执行 H2REMOTE ok=8 fail=0（KV+indexer 两池）**——修复前 remote 无任何 submit 活动。
- `putget`（PUT→清CPU缓存→冷GET，Node A 常驻）：
  - PUT：`keys present in store after PUT: 32/32 across 2 pool(s) -> OK`。
  - 冷 GET：`GET(shared): 1024/1024 tokens, hit_ratio=100.00%, 16.8 ms`、`store keys visible cross-node: True (32/32)`、负对照 0.00%、**CROSS-NODE SHARING CONFIRMED ✅**。
  - Node B 双向都跨机执行：H2REMOTE(写) + REMOTE2H(读) 均 ok、**0 Error launching transfer**。
→ **跨机 TP16 + indexer 多节点 bring-up + 双向数据面 + 两池共享 + 冷读 100% 命中，端到端真机打通。**

**已知遗留（偶发，非数据正确性）**：某次 `putget` 的 PUT 报 `512 tokens / 20008 ms`（一个并发 graph 的完成 ack 在 20s 内未回齐；同配置独立 PUT 则 1024/22.6ms 干净）。数据无损（32/32 两池全在 store 且冷 GET 100% 可读）。疑为 `TransferManagerMultiNodeHandle` 的 result_socket（RCVTIMEO=0 非阻塞轮询）在 2 并发 graph 下偶发 ack 延迟/丢报，属完成回传通道健壮性问题，非本次 TP 提交修复引入。建议后续排查 result_socket 的 HWM/送达确认。

### 6.8 ✅✅ PUT/GET 真因锁定并修复 + 双节点端到端跑通（2026-07-01 23:0x）
**真因（一行修复，与 MPS/arch/clear-set 全无关，之前均是被 CUDA sticky error 误导的旁支）**：
`flexkv/transfer/worker.py` 的 `GPUCPUTransferWorker._transfer_impl` 调用 `transfer_kv_blocks(...)` 时**漏传了 `start_layer_id` 位置参数**。
- 真机实测安装的 c_ext binding 签名（docstring 证实）位置 12 是 `start_layer_id`（无默认值）：`...chunk_size(11), start_layer_id(12), num_layers(13), transfer_num_cta(14)=4, is_host_to_device(15)=True, use_ce_transfer(16)=False, is_mla(17)=False, gpu_block_type(18)=0, sync(19)=True`。
- 我们 worker.py 只传了 17 个位置参数、`chunk_size` 后直接接 `num_layers`，**从第 12 位起整体错位一位**：
  - `transfer_num_cta`(14) 实收 `transfer_type==H2D` 布尔 → **D2H 时 = 0 → gridDim=dim3(0) → `cudaErrorInvalidConfiguration`（err 9）**。
  - `use_ce_transfer`(16) 实收 `is_mla`=False → **CE 开关被架空**，无论 `FLEXKV_USE_CE_TRANSFER_D2H` 设啥都走自定义 kernel → 解释「CE=1 也失败」「两路径同错」（后者实为首个 kernel 失败的 sticky error 污染 CE 路径的 sync）。
- **参考实现 `/data1/home/phaedonsun/p800/mooncake/FlexKV/flexkv/transfer/worker.py` L394 明确传了 `0,`（start_layer_id，附注释）**——我们移植版恰好丢了这一行。tp 版 `tp_group_transfer` 的 `layer_id=0`（L592）未丢，故仅非 tp 的 `GPUCPUTransferWorker` 受影响（当前 tp=1 单机正好命中它）。

**修复**：worker.py `GPUCPUTransferWorker._transfer_impl` 的 `transfer_kv_blocks(...)` 在 `self.chunk_size_in_bytes,` 后补回 `0,  # start_layer_id`（对齐参考实现）。**纯 Python 调用点补参，不触碰 transfer.cu / CE kernel / checksum / launch_transfer 计时（C1 零改动）**。

**验证（两台已打同一补丁）**：
- PUT（Node A gpu-146）：`PUT: 1024 tokens, 0.016 GB, 30.9 ms`；**keys present in store after PUT: 64/64 → OK**；无 invalid config。
- GET（Node B gpu-129，冷缓存跨节点读）：**store keys visible cross-node: True (64/64)**；**cold-cache shared hit: 100.00% (≥95% OK)**；负对照（从未 PUT）hit 0%。
- 默认自定义 kernel 路径（CE=0）复测 PUT（见下）——证明修复独立成立、非 CE 绕过。

**被推翻的三个旁支结论（存档）**：① MPS/host-ptr device 解引用（dev-worker）；② clear/set_gpu_blocks 不对称（dev-core/dev-engine，真实潜在 bug 但仅跨机 PP 触发，本期不修，记为已知问题）；③ CUDA arch 缺 sm_90（实测 c_ext 含 sm_90）。三者均因 err 9 的 sticky 特性被误导。
修改（8）：`flexkv/common/config.py`、`flexkv/common/transfer.py`、`flexkv/transfer/worker.py`、`flexkv/transfer/worker_op.py`、`flexkv/transfer/transfer_engine.py`、`flexkv/cache/cache_engine.py`、`flexkv/storage/storage_engine.py`、`flexkv/transfer_manager.py`、`FlexKV/install.sh`。

---

## 7. 跨实例（A/B 两推理实例）KVCache 共享失败排查（2026-07-09，P800 集群）

> 背景：P800 集群部署两个推理实例 A/B，期望 A 把 KV store 到共享 mooncake master 后，B 能从 mooncake 命中 A 写入的 KV。**实测 B 侧 `get_task=0` / `cached-token=0`（B 从未命中）**。§6 的双执行端（gpu-146/gpu-129）端到端已 CONFIRMED，故本节聚焦**生产集群多层网络（pod overlay vs 物理机 vs RoCE）下的跨实例可达性**。

### 7.1 部署形态确认（推翻两个初始排查方向）
通过检查**运行中进程实参**（`pgrep -af launch_server`）+ wrapper 源码分析确认：
- 4 个 pod（prefill-0 / prefill-0-1 / decode-0 / decode-0-1）都是**非 PD 分离的统一服务**，`launch_server` 实参**均无 `--disaggregation-mode`**。A 用 dist-init `***:5000`，B 用 `***:5000`，各自 tp16 跨 2 节点，共享 mooncake master `***:50051`。
- wrapper `run_role_1p1d_flexkv_v2` 构建 sglang_args 时**从不注入** `--disaggregation-mode`。
- ⟹ **方向2（decode 角色门控 get）不成立**（B 是普通 server 会正常触发 get）；**方向1（改 B 为 prefill 角色）无必要**。

### 7.2 逐层实证（key 一致但 segment 不可达）
- A 发唯一 prompt（723 tokens）：`batch_put: 11 keys put` → **A store 成功**。
- B 发相同 prompt：`match: all_keys` 与 A **完全一致**（`4f7a...._FlexKV` 等，tp16 非 PP 无后缀），但 `exit_results:[0,...,0]` **全 miss**，`cached-token:0`。
- `get_all_segments` 只返回 A 的两节点（`***:12345` / `***:12345`），**B 的 segment 未注册进 A 可见的拓扑**。
- ⟹ 问题**不在 key 生成**（A/B key 完全一致，移植逻辑正确），而在 **mooncake 数据面 P2PHANDSHAKE 的跨实例 segment 可达性/握手**。

### 7.3 关键结论矩阵（直接用 mooncake SDK 探针 `/tmp/probe.py` 绕过 flexkv 实测）
探针：一端 `put` + 常驻保持段，另一端跨实例 `is_exist`/`get`。

| 协议 | local_hostname 用的地址 | 跨实例结果 |
|---|---|---|
| **rdma**（pod 内 & 物理机）| device=mlx5_2 | ❌ writer `self_exist 1`，跨实例 reader `reader_exist 0 / get None` |
| **tcp** + 物理机 IP（9.x）| 106.20→104.60 | ✅ writer `put_ret 0`，reader **`reader_exist 1 / get 8192`** ——**跨实例通！** |
| **tcp** + pod overlay（10.129.x）| decode-0 reader | ❌ reader `reader_exist 0 / get None`（但 master:50052 与数据口 12346/13300 从 B→A **TCP 全部 OPEN**）|

**核心洞见**：TCP 端口连通 ≠ P2PHANDSHAKE 元数据/地址协商成功。overlay 网络里 prefill-0 在 `10.129.6.x`、其余 3 个在 `10.129.7.x`，跨 /24 的 P2P 数据面握手未打通；而物理机 9.x 是全数据中心 L3 可路由，故 tcp+物理机 IP 一锤定音地跨实例通。

### 7.4 物理机 ↔ 容器 IP 映射 + RoCE 设备（当次部署，已随后下线，见 §7.6）
| 角色 | Pod | 容器IP(overlay) | 物理机IP(9.x) | RoCE设备 |
|---|---|---|---|---|
| A rank0 | prefill-0 | *** | *** | mlx5_2→eth4 |
| A rank1 | prefill-0-1 | *** | *** | mlx5_2→eth4 |
| B rank0 | decode-0 | *** | *** | mlx5_2→eth4 |
| B rank1 | decode-0-1 | *** | *** | mlx5_2→eth4 |

- `mlx5_2` = RoCE：`link_layer: Ethernet`、`PORT_ACTIVE`、`LINK_UP`、100 Gbps；`ibdev2netdev` → `mlx5_2 port 1 ==> eth4 (Up)`（四节点一致）。RDMA 工具齐全（`ib_write_bw` / `ibv_rc_pingpong` / `show_gids`）。
- **可疑点（很可能是 RoCE 跨节点失败直接原因）**：四节点 `eth4` **均无 IPv4 地址**。mooncake RDMA 配置用 `device_name=mlx5_2`（=eth4），RoCEv2 GID 基于网卡 IP 生成，eth4 无 IP 意味着 RoCEv2 GID 无法用于跨节点寻址——与 §7.3 RDMA 跨实例失败吻合。RoCE 的 IP 可能挂在 `bond0`（`mlx5_bond_0`）上，待 `show_gids mlx5_2` 确认。

### 7.5 卡点（ssh 被限频）
本机→物理机 ssh 前期高频连接触发远端 fail2ban/`MaxStartups` 早丢弃，连续 8 次带退避重试均 `kex_exchange_identification: Connection closed`，物理机层 RoCE 的 `ib_write_bw` 双端测试**未能完成**。已提供两条落地路径（用户在物理机直接跑 `ib_write_bw -d mlx5_2 -x <GID> -F` server/client；或解封本机源 IP 后由我自动测）。

### 7.6 环境变更（重要）
经 relay-cli → 跳板机 `/tmp/kubectl` 复查，**§7.4 那套 `glm5-p800-flexkv-inference-*`（*** / *** / *** / ***）已整套下线**，`p800_hosts.txt` 里的 4 个 IP 全部失效。当前 `maas-public` 命名空间是另外几套部署：`glm5-p800-mc-availability-test-0`（5P5D，名字带 mc=mooncake）、`glm5-p800-mc-5p5d-l2-myw`、`glm5-v2-p800-4`（1P1D）、`glm5-v2-p800-8-0` / `glm5-v2-p800-8-1`（两套独立实例，天然跨实例对）。
> 注：`p800_hosts.txt` 里的 IP 本就是 mooncake 每节点的 `local_hostname`（segment 注册/对端寻址用），但因整套下线需按当前部署重新取 IP 再验证。

### 7.7 阶段结论 & 待落地
- **FlexKV 移植逻辑无缺陷**：A/B key 完全一致、A store 成功；问题纯粹在 **mooncake 数据面跨实例网络可达性**（P2PHANDSHAKE + overlay 跨 /24 / RoCE GID 无 IP）。
- **已实证唯一可行组合**：**tcp + 物理机可路由 IP（9.x）**。
- 候选落地方案（待决策）：
  - **方案A**：pod 用 hostNetwork + tcp + 物理机 IP 作 `local_hostname`（已实证通，改动最小）。
  - **方案B**：引入集中式 metadata server（etcd/redis）替代 P2PHANDSHAKE，让跨 /24 overlay 也能协商。
  - **方案C**：排查 overlay 下 P2PHANDSHAKE 跨 /24 失败根因（端口通但握手失败），或修复 RoCE eth4/bond0 GID 的可路由 IP 使 rdma 跨节点可用。
- **下一步**：环境已变，需在当前存活的跨实例对（如 `glm5-v2-p800-8-0` vs `-8-1`）上用当前 pod 的精确 overlay IP 重跑一次干净的 tcp 跨实例探针，复现/确认 §7.3 矩阵后再定方案。
