# P800 5P1D（sglang + FlexKV + mooncake-store）12 台节点分配与角色记录

更新时间：2026-07-09 16:10

> 目标：在专属 20 台（见 `doc/p800_flexkv_20nodes.md` / `p800_hosts.txt`）里分配 **12 台**做 5P1D 测试：
> 5 个 prefill 实例 + 1 个 decode 实例，每个实例 tp16 跨 2 节点（P800 每台 8 卡）。
> **master 节点、decode 节点都严格限定在这 20 台内**。

---

## 1. 每实例并行度（来自 `zittozhang_scripts/scripts/decode_start_1p1d_flexkv_v2.sh`）
- `--tp-size 16 --ep-size 16 --dp-size 1`，tp16 = 16 卡 = **2 台 P800**（8 卡/台）。
- 每实例 rank0 为 `dist-init-addr`（`:5000`），rank1 连 rank0；实例内 tp 通信走 **BKCL RDMA（eth1~eth4）**，与 mooncake 无关。
- 跨实例 KVCache 共享走 **共享的同一个 mooncake master**（各节点把本地 segment 注册进 master，靠 mooncake 数据面互访）。

## 2. 12 台分配表（严格取自 20 台名单）

| 实例 | 角色 | rank0（dist-init / leader）| rank1（worker）| 备注 |
|---|---|---|---|---|
| **P0** | prefill | *** | *** | rank0 兼 **mooncake master**（`:50051`）|
| **P1** | prefill | *** | *** | |
| **P2** | prefill | *** | *** | |
| **P3** | prefill | *** | *** | |
| **P4** | prefill | *** | *** | ⚠️ 唯一落在 `10.129.0.x` 段 |
| **D0** | decode | *** | *** | |

- **master 节点**：`***:50051`（复用 P0 rank0，不额外占机）。
- **decode 节点**：`***` / `***`（均在 20 台内）。
- 合计 **12 台**，全部来自 `p800_hosts.txt`。

### 分配依据（/24 网段与 §7 教训）
20 台按 /24 分布：`0.x`(6) / `1.x`(10) / `3.x`(2) / `5.x`(2)。
`doc/mooncake_store_impl_status.md` §7 结论：**mooncake 数据面跨 /24 的 P2PHANDSHAKE 是高风险点**（overlay 跨 /24 端口通但握手失败）。因此：
- 把 **P0~P3 + D0（5 实例，10 台）全部聚在 `10.129.1.x` 同一 /24**（KV 生产/消费主路径尽量同段，降低跨段风险）。
- 仅 **P4 落在 `10.129.0.x`**（`1.x` 只有 10 台，第 6 个实例必须跨段），作为**跨 /24 共享的真实压力样本**。
- master 放 `1.x`（多数实例同段），减少 master 与 worker 间跨段。

## 3. 备用 8 台（未分配，保留）
`***`、`***`、`***`、`***`、`***`、`***`、`***`、`***`

## 4. 探针验证计划（先于铺开 5P1D）
先用 2 台跑 mooncake SDK 探针，验证这批 overlay 的数据面可达性，**再决定 5P1D 用 rdma 还是 tcp**：
- **writer + master**：`***`（1.x 段，master 所在）
- **reader**：`***`（0.x 段，即 P4 rank0）——**跨 /24**，直接复现 §7 最高风险边界。
- 判定：reader `is_exist=1 / get=<len>` ⟹ 跨 /24 数据面通，5P1D 可用 rdma；否则需退回 tcp + 可路由 IP（见 §7.3 矩阵）。
- 探针资产：`zittozhang_scripts/pods/5p1d_probe/`（2 个 pod YAML）+ `zittozhang_scripts/scripts/mooncake_probe.py`。

## 5. 探针实测结果（2026-07-09 16:40，已完成，真实）
writer+master=`***`(1.x)、reader=`***`(0.x)，**跨 /24**，master `***:50051`：

| 协议 | writer | reader（跨 /24）| 结论 |
|---|---|---|---|
| **rdma**（device=mlx5_2）| setup=0 / put=0 / self_exist=1 | **reader_exist=1 / get_len=8192** | ✅ **PASS** |

**核心结论（推翻 §7 悲观判断）**：这批新 20 台上 **rdma + overlay 跨 /24 直接打通**（新节点 RoCE GID 配置正常，探针里 GID 为 IPv4-mapped）。⟹ **5P1D 用 `protocol=rdma`**（比 tcp 快，且脚本默认即 rdma）。

## 6. 环境事实（部署前必读，实测得出）
镜像 `sglang_p800_glm5_int8-2026-05-22`：
- conda env 只有 **base(py3.13)** 与 **python310_torch25_cuda(py3.10.12)**；**无 `flexkv_env`**。
- `python310_torch25_cuda` 有 `sglang 0.5.6`，但**无 flexkv、无 mooncake**。
- **无 `mooncake` / 无 `mooncake_master`**（需装 whl）。
- 登录 shell 的 `python3` = base py3.13（**不是** sglang 所在 env）。

**关键依赖链**：
1. sglang 启动**必须经 wrapper** `run_role_1p1d_flexkv_v2`（跳板机 `/root/zittozhang/scripts/`）——它 `conda activate flexkv_env` 并设置全部 P800/BKCL/XSGL/XMLIR/NSA env。直接 `python3 -m sglang.launch_server` 会漏掉这些 env，**不可用**。
2. `flexkv_env` **不在镜像里**，由 `init_env_v2.sh`（跳板机 `/root/zittozhang/scripts/`，另有 `_all`/`_prefill_only` 变体）创建：`conda create flexkv_env --clone python310_torch25_cuda` → 需 pod 内 `/workspace/flexkv_dev` 源码 → `./build.sh` 编译 FlexKV → `/workspace/sglang_dev` pip editable → 装 xflashinfer_ops whl。
3. **mooncake 要装进 `flexkv_env`**（wrapper 用的 env）。用跳板机 whl：`mooncake_transfer_engine-0.3.10.post2+...-cp310-...whl`（cp310 匹配）+ `pip install --no-deps`。装后提供 `mooncake.store` 与 `/usr/local/bin/mooncake_master`（探针在 python310 env 已验证可用）。
4. wrapper 契约（可被 env 覆盖，供 5P1D 多实例用）：`PD_ROLE_OVERRIDE` / `RANK_OVERRIDE`(0|1) / `WORLD_SIZE`(=2) / `MASTER_ADDR`(本实例 rank0 IP，dist-init) / `PREFILL_MASTER_ADDR_LIST` / `DECODE_MASTER_ADDR_LIST` / `ROUTER_COUNT=-1`(关 router) / `AUTO_IB_DEVICE=false`；wrapper 自动注入 `--nnodes/--node-rank/--dist-init-addr`（故 node 脚本传的 sglang 参数里**不要**再带这三个）。

## 7. 修正后的 5P1D 部署流程
1. **apply 12 pod**：`kubectl apply --validate=false -f pods/5p1d/*.yaml`（`nodeSelector` 的 `null` 会被客户端校验拒绝，且会导致 NodeAffinity 不可调度——已去掉 null，仅留 3 个等值标签；`--validate=false` 仍建议）。pod 需**挂载 `/workspace`**（flexkv_dev/sglang_dev 源码所在）——生成器已补。
2. **建 flexkv_env**：每 pod 跑 `init_env_v2.sh` 编译 FlexKV（前提：节点 `/workspace/flexkv_dev` + `/workspace/sglang_dev` 源码就位；新节点若无需先分发）。
3. **装 mooncake**：每 pod `flexkv_env` 里 `pip install --no-deps <whl>`。
4. **分发 wrapper + node 脚本**：`run_role_1p1d_flexkv_v2`、`nonpd_start_5p1d_node.sh` → 各 pod `/workspace/zittozhang/`。
5. **起 master**：master 节点 pod `mooncake_master --port 50051 --metrics_port 9003`。
6. **错峰拉起 5P + 1D**：`start_5p1d_mooncake.sh`（PROTOCOL=rdma）。
7. **验证**：`/health` → 发唯一 prompt 到某 P → 另一实例/decode 命中共享 KV（cached-token>0）。

## 8b. 1P1D 端到端 bring-up 进展（2026-07-09 17:xx，方案A）
按方案A先起 **1P1D（P0 两节点 `***`/`***` + D0 两节点 `***`/`***`，共 32 XPU）**验证：
- ✅ apply 4 pod（`--validate=false`）；prefill-0 首次 `UnexpectedAdmissionError`（cpu:234 与 probe 冲突），删探针 pod 释放 CPU 后重建成功。
- ✅ `init_env_5p1d.sh`（由 `init_env_v2.sh` 改 pod 过滤为 `glm5-p800-5p1d-`）在 4 pod **编译 flexkv_env 全部成功**（conda clone + build.sh + sglang editable + xTriton + xflashinfer；串行约 25 分钟）。
- ✅ 4 pod 的 flexkv_env `pip install --no-deps` 装 mooncake whl，`import flexkv,sglang,mooncake` 全部 `VERIFY_OK 0.5.6`。
- ✅ 分发 wrapper `run_role_1p1d_flexkv_v2` + `nonpd_start_5p1d_node.sh` 到各 pod `/workspace/zittozhang/`。
- ✅ prefill-0 起 mooncake master（flexkv_env 内 `mooncake_master --port 50051 --metrics_port 9003`，serving）。
- ✅ 4 节点经 wrapper 拉起 sglang（注意：nohup 外层重定向需先 `mkdir -p logs`），**正在加载 GLM-5 权重**（152 shards）。
- ⏳ 待：health 就绪 → 跨实例 KV 共享验证（P store / D 命中）。

**经验教训**：① `nodeSelector` 的 `null` 会致 NodeAffinity 不可调度，去掉仅留 3 等值标签；② apply 用 `--validate=false`；③ pod cpu:234 高请求，同节点勿并存其他重 pod；④ mooncake 装 `flexkv_env`（wrapper 用的 env），非 python310；⑤ 跳板机 relay 黑名单禁 `bash -c`，用 `sh -c` 或 `bash <script>`；⑥ 外层 nohup 重定向前先建 logs 目录。

## 8b. 【重大方向纠正】必须做 PD 分离，而非非 PD 统一服务（2026-07-09 17:55）
之前误按 `decode_start_1p1d_flexkv_v2.sh`（非 PD 统一服务）搭建：每个节点同时带 `--kv-connector-cls flexkv` + `--speculative-algorithm EAGLE`，warmup prefill 前缀匹配时 EAGLE 使 radix key 变 bigram(2D)，FlexKV `assert token_ids.ndim==1` 崩溃。

**用户澄清：要的是真 PD 分离。** 对照 PD 参考脚本，正确架构为：

| | Prefill (`prefill_start_flexkv.sh`) | Decode (`decode_start_flexkv.sh`) |
|---|---|---|
| `--disaggregation-mode` | prefill | decode |
| FlexKV connector | ✅ `--kv-connector-cls flexkv` | ❌ 不带 |
| radix cache | 有（+flexkv 前缀缓存） | ❌ `--disable-radix-cache` |
| EAGLE 投机 | ❌ 无 | ✅ 有 |
| dp / attention | dp1、disable-cuda-graph、nsa-cp | dp16、dp-attention、dp-lm-head、round-robin |

⟹ **EAGLE 是 decode 专属、FlexKV connector 是 prefill 专属，二者在不同节点/不同配置，天然不冲突**（decode 的 EAGLE bigram 不会碰 FlexKV）。之前崩溃纯粹是"非 PD 统一服务把两者塞进同一节点"造成的。

### 正确的 PD 5P1D 技术栈（改用）
- 编排：`start_5P1D_flexkv.sh`（跳板机 `/root/zittozhang/scripts/`；分发 + 错峰拉起 5P+1D）。
- wrapper：**`run_role_flexkv`**（PD 版，非 `run_role_1p1d_flexkv_v2`）——从 hostname 推断 P/D 角色、注入 `--disaggregation-*`、prefill rank0 起 `sglang_router --pd-disaggregation`。
- 单节点：`prefill_start_flexkv.sh`（5 个 P）、`decode_start_flexkv.sh`（1 个 D）。
- KV 流：P→D 走 `--disaggregation-transfer-backend`（mooncake transfer engine）；FlexKV 在 prefill 侧做 KV 前缀缓存/offload。
- 复用：已建好的 4 pod flexkv_env 编译成果可继续用（同 env）。

### 待办（executor 恢复后）
1. 读 `run_role_flexkv` 确认：角色/rank/master-list 推断、`--disaggregation-transfer-backend`/`-ib-device`/bootstrap 端口、mooncake 接入方式。
2. 先做 **1P1D PD 验证**：P0（prefill_start_flexkv.sh）+ D0（decode_start_flexkv.sh），跑通 P→D KV 传输与端到端推理。
3. 通过后按 `start_5P1D_flexkv.sh` 铺满 5P1D（pod 命名需匹配 wrapper 的 hostname 推断规则）。
4. 之前为非 PD 生成的 `pods/5p1d/*.yaml`、`nonpd_start_5p1d_node.sh` 需按 PD 重做/替换。

## 8c. 真 PD 分离 1P1D bring-up（2026-07-09 19:xx，进行中）
改用真 PD 分离栈（wrapper `run_role_flexkv` + `pd_start_5p1d_node.sh`）重跑 1P1D：
- ✅ **精确清理旧非 PD 残留**：`cleanup_5p1d_procs.sh`（TERM→KILL 清 sglang/run_role/router，[x] 正则；补刀清孤儿 run_role）——4 pod 的 sglang/run_role/router 全清 0，**mooncake master 完好保留**（prefill-0 pid 1970/1972，:50051 LISTEN）。
  - ⚠️ 教训：容器内 `pkill`/`pgrep` 是 **busybox（`-f` 子串匹配）**，`[x]` 正则技巧对它失效（承载 sh 的 cmdline 含 pattern 字面子串→自杀，exit 143）；清残留改用**拼接 pattern**（`"run_role""_1p1d"`）使承载 sh cmdline 不含完整串。
- ✅ **参数对照**：`pd_start_5p1d_node.sh` 的 prefill/decode 两分支 sglang 参数 + FLEXKV env 与参考 `prefill_start_flexkv.sh`/`decode_start_flexkv.sh` **逐行一致**。PD 下 EAGLE 只在 decode（且 decode 无 flexkv connector + `--disable-radix-cache`），prefill 有 flexkv connector 但无 EAGLE ⟹ 天然不冲突。
- ✅ **修复 config bug**：`MooncakeStoreConfig.from_file` 对 JSON 的 `global_segment_size` 会自乘 1024³（源码 `mooncake_store_utils.py:109-113`），故 JSON 里应填 **GB 数值**（原脚本误填字节数，会段分配爆炸）。已改为填 GB。其余字段名（master_addr/metadata_server/protocol/device_name/local_hostname/enable_ssd_offload/ssd_offload_path/master_metrics_port）与 from_file **完全一致**；env 名 `FLEXKV_USE_MOONCAKE_STORE_BACKEND`/`FLEXKV_MOONCAKE_STORE_CONFIG_PATH` 正确（`config.py:410-419`，`bool(int(...))`）。
- ✅ **5 prefill 共享同一 master 前缀缓存**：prefill 分支启用 `FLEXKV_USE_MOONCAKE_STORE_BACKEND=1` + config 指向同一 `MOONCAKE_MASTER_ADDR`（本轮 `***:50051`）。数据面：本轮探针已实证新 20 台 **rdma 跨 /24 PASS**（§5），故 `protocol=rdma`。
- ✅ **1P1D 启动**：`pd_launch_1p1d.sh` 按角色注入 env（P0 rank0/1=`***`/`***`，D0 rank0/1=`***`/`***`；master=`***:50051`）。**关键**：`PREFILL_MASTER_ADDR_LIST`/`DECODE_MASTER_ADDR_LIST` 只列 1P1D 单实例（不能用 5P 默认，否则 router 按 5 实例等待）。4 pod 经 wrapper 拉起，**正在加载 GLM-5 权重**。
- ⏳ 待：health 就绪 → P→D disaggregation KV 传输 → 端到端推理 → prefill 侧 FlexKV+mooncake 前缀缓存命中验证。

## 8d. 【关键发现】PD 端到端已通，但 FlexKV mooncake-store 前缀缓存实为未部署（2026-07-09 19:2x）
1P1D PD 端到端**已跑通**（router:8501 → P0 prefill → disaggregation KV 直传 → D0 decode → 正常出 token，`prompt=13/completion=64`，无崩溃）。**但这是纯 sglang PD 分离 + mooncake transfer engine 直传 KV**，FlexKV 侧的 mooncake-store 共享前缀缓存**并未生效**：
- 实测 prefill FlexKV `cache_config`：`enable_remote=False`、无 `use_mooncake_store_backend` 字段。
- 根因（已核实）：**pod `/workspace/flexkv_dev` 与跳板机 `/root/zittozhang/FlexKV` 都是不含 mooncake-store 移植的旧版 FlexKV**（`grep use_mooncake_store_backend`=0、无 `flexkv/external/` 目录）。
- mooncake-store 移植（11 文件：`external/{__init__,mooncake_store_keys,mooncake_store_utils}.py` + `config.py`/`common/transfer.py`/`worker_op.py`/`transfer_engine.py`/`cache_engine.py`/`storage_engine.py`/`transfer_manager.py`/`worker.py` + `install.sh`）**只在本地 `/data1/.../FlexKV`**——见 `doc/mooncake_store_impl_status.md`（70 单测 + H20 双节点 CONFIRMED，**P800 真机未端到端验证**）。
- 附带修复（本地 `config.py`）：`__post_init__` 的 `enable_remote` 派生原漏 `use_mooncake_store_backend`（在 env 更新前就定为 False）→ 已改为在 env 更新后 `enable_remote = enable_3rd_remote or use_mooncake_store_backend`（对齐 `make_cache_config`）。否则 sglang `flexkv_connector._prefetch_enabled` 恒 False。

### 决策点（待用户拍板）
- **方案①（已达成，稳）**：接受"PD 分离 + mooncake transfer 直传 KV"形态（P→D 走 disaggregation，无 FlexKV 跨实例前缀缓存），直接铺开 5P1D。
- **方案②（实现共享前缀缓存，需部署移植版）**：把本地含 mooncake-store 移植的 FlexKV 部署到 pod `/workspace/flexkv_dev`（Python 为主，editable 即时生效；c_ext 已编译；mooncake.store 已由 whl 提供）→ 重启 prefill → 验证 `enable_remote=True`/`Store initialised`/跨 prefill 前缀命中。**风险**：P800 真机首次跑移植代码，需先在 1P 验证再铺开。

## 8e. 【重大成功】新版 FlexKV(mooncake-store) 部署 + 前缀缓存激活（2026-07-09 20:0x）
用户提供跳板机 `/root/phaedonsun/FlexKV`（较新、含 mooncake-store 移植：`external/` + `use_mooncake_store_backend`）。把修复 patch 进去、打包分发到 2 个 prefill pod、重编译安装、重启，**FlexKV mooncake-store 前缀缓存终于真正激活**：
```
enable_remote=True; use_mooncake_store_backend=True
[MooncakeStoreConfig] global_segment_size ... 8 GB
[MooncakeStoreClient] Store initialised: master_addr=***:50051, protocol=rdma
[MooncakeStoreCacheEngine] pools=['FlexKV','FlexKV_indexer'], hit_required=[...]
```

### 部署流程（可复用脚本，均在 zittozhang_scripts/scripts/）
1. `patch_mooncake_enable_remote.py`：patch `CacheConfig.__post_init__` 的 `enable_remote` 派生（在 env 更新后、算上 mooncake）。
2. `patch_userconfig_mooncake.py`：patch **`UserConfig.__post_init__` 读 env**（根因修复，见下）。
3. `deploy_flexkv_mooncake.sh`：打包(排除 .git/.so) → 分发 → 恢复 xxHash → `build.sh`（cmake 编 c_ext + `pip install -e`）。**注意**：build 前必须 `export PATH=flexkv_env/bin:$PATH`（否则用 base py3.13 装失败）；build 后 pip 会拉 PyPI triton 3.1.0，需 `fix_triton_verify_flexkv.sh` 从 python310 env 恢复 **xTriton 3.0.0**。
4. `hard_restart_prefill_mooncake.sh`：按 `/proc/pid/exe` 彻底 kill flexkv_env python（保留 mooncake_master）→ 重起 master → 重启 prefill（清彻底，否则残留 tp16 worker 占 GPU 致 `memory capacity unbalanced`）。

### 三处根因（层层深入，均已修复）
1. `CacheConfig.__post_init__` 的 `enable_remote` 派生漏 mooncake（在 env 更新前定为 False）→ 已 patch。
2. **最终根因**：sglang 经 `flexkv.integration.config.FlexKVConfig.from_env()` 用 `user_config=field(default_factory=UserConfig)` 默认构造，`from_env()` 不设 mooncake 字段；而 `UserConfig.__post_init__` **原本不读 env** → `use_mooncake_store_backend=False` → `make_cache_config`(config.py:649) 用它覆盖 CacheConfig → enable_remote 归 False。⟹ 修 `UserConfig.__post_init__` 读 env（对齐 CacheConfig）后 `FlexKVConfig.from_env().user_config.use_mooncake=True`。
3. sglang 侧完全不感知 mooncake（grep 无引用）——**无需改 sglang**，仅靠 flexkv 两个 `__post_init__` 读 env 即透明接入。

### 待续（relay JWT 过期阻塞）
- 端到端推理 + 验证 KV 写入共享 mooncake（`batch_put keys`）+ 跨 prefill 前缀命中（`cached-token>0`）。
- 铺开 5P1D：其余 3 个 prefill（P1/P2/P3/P4）同法部署新版 FlexKV。

## 8. 当前阻塞 / 下一步
- **已验证**：数据面 rdma 跨 /24 通（§5）；镜像无 flexkv_env、无 mooncake、无 `/workspace/flexkv_dev`/`sglang_dev`（同镜像新 pod 实测均无——旧 pod 里的源码是当初手动放进容器可写层、未持久化）。
- **核心阻塞（建 flexkv_env 的源码）**，两条路径二选一：
  - **路径A（干净，慢）**：向 12 pod 分发 FlexKV 源码（跳板机 `/root/phaedonsun/FlexKV.tar.gz`）+ sglang 源码 → 每 pod 跑 `init_env_v2.sh` 编译（`build.sh` 逐 pod 编译，较慢）+ 装 mooncake whl。
  - **路径B（快，需验证可迁移性）**：从旧 pod（`glm5-p800-flexkv-inference-prefill-0`，已有编译好的 `flexkv_env`）打包 `/root/miniconda/envs/flexkv_env` → 分发解压到 12 新 pod 同路径（路径一致，conda clone 可用）+ 装 mooncake whl。省去 12 次编译。
- **资源提醒**：12 pod × 8 XPU = 96 XPU；建议先起 1 对 P（2 pod）跑通端到端，再铺满 5P1D。
- 探针 pod（`mooncake-probe-writer/reader`，占 2 台但无 XPU 请求）验证完，可 `kubectl delete` 清理。
- **产物**：`pods/5p1d/*.yaml`(12)、`scripts/gen_5p1d_pods.sh`、`scripts/nonpd_start_5p1d_node.sh`（走 wrapper）、`scripts/start_5p1d_mooncake.sh`、`scripts/mooncake_probe.py`。
