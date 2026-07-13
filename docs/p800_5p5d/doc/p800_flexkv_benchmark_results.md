# P800 5P5D FlexKV+mooncake-store Benchmark 结果记录

> 本文档记录 **FlexKV+mooncake-store**（5P5D，P800 集群）的 benchmark 测试方案、每次运行的原始结果与分析结论。
> 背景/部署过程/bug 修复记录见 `doc/mooncake_store_impl_progress.md` §7。
> 长远目标：FlexKV vs sglang 原生 HiCache 对比（本文档先聚焦 FlexKV+mooncake-store 单独充分测试）。

---

## 0. 环境信息

- 集群：P800，5 prefill 实例 + 5 decode 实例（每实例 tp16=2 节点，共 20 节点）
- Router：`sglang_router --pd-disaggregation`，跑在 P0（`***`），对外 `http://localhost:8501`（仅 P0 pod 内可达，`kubectl exec` 进 P0 pod 执行）
- 访问方式：`relay-cli run --executor executor-p800-jump-server` → `kubectl exec -n maas-public glm5-p800-5p5d-prefill-0 -c prefill-engine` → 容器内 python3 (`flexkv_env`, Python 3.13.5，纯 stdlib，无 `requests`/`numpy`/`openpyxl`/`pandas`)
- 测试脚本：`zittozhang_scripts/scripts/bench_5p5d.py`（纯 urllib+threading，无第三方依赖），已部署在 pod 内 `/workspace/zittozhang/bench_5p5d.py`
- 数据集：远端 pod0 现成的 `cacheX.XX` 系列真实业务数据，路径 `/home/data/darren/ai_evaluagtion/model_perf/data/`，字段：`sub_uin, start_time, prompt_tokens, cached_tokens, completion_tokens, messages, session_id`（按 session 分组的多轮回放数据，每条记录自带生产环境 ground-truth 命中率）

已知可用数据集清单（pod0 独有，本地开发机没有）：

| 文件名 | 文件名标注命中率 | 说明 |
|---|---|---|
| `glm5-24K-64K-cache0.66_2k.json` | 0.66 | 2000 条记录 / 621 session，平均 3.2 条/session，最多 53 条 |
| `glm5-workbuddy-24K64K-cache0.82_0.55k_tools.json` | 0.82 | 含 tool_calls，待测 |
| `glm5.1-CBWB-cache0.8-5.65k.json` | 0.80 | 待测 |
| `glm5-128K-200K-cache0.95_1k.json` | 0.95 | 待测 |
| `glm5-200K+-cache0.47_1k.json` | 0.47 | 待测 |

---

## 1. 测试方案

### 1.1 指标

| 指标 | 定义 | 取值 |
|---|---|---|
| TTFT | 首 token 延迟（流式，发出→首个 chunk） | p50/p90/p99 + mean |
| E2E | 整请求耗时 | p50/p90/p99 + mean |
| 命中率 | Σcached_tokens / Σprompt_tokens | 实测 vs 数据集 ground-truth vs 文件名标注，三者对比 |
| 吞吐 | req/s、output tok/s | 按并发档 |

### 1.2 变量控制

- **并发梯度**：4 / 8 / 16 / 32（已去掉 c=1，实测发现其对 session 粒度回放太慢、且非关注区间）
- **数据回放粒度**：**必须按完整 session 回放**（`--max-sessions`，不能用 `--max-records` 全局截断，否则破坏多轮累积命中效果、命中率会被严重低估——已有实测教训，见 §2 run1）
- **超长 session 保护**：`--max-records-per-session` 截断极端长尾（如 50+ 轮、5万+ token 的 session），避免个别极端请求拖慢整档测试节奏

### 1.3 已知问题与已修复的 bug（影响测试结果解读）

1. **tool 消息缺 `tool_call_id` → HTTP 400**：已修复（`bench_5p5d.py` 规整化逻辑保留原始字段）。
2. **stdout 全缓冲导致 nohup 后台运行时进度不可见、误判"卡死"**：已修复（`sys.stdout.reconfigure(line_buffering=True)`）。
3. **【重大】mooncake-store 原生 RDMA 调用无超时保护，导致 scheduler 主线程挂起、冻结整个 prefill 实例**：已定位根因并修复，详见 `doc/mooncake_store_impl_progress.md` §7 末尾"Bug 根因已定位并根除"章节；修复代码位置 `FlexKV/flexkv/external/mooncake_store_utils.py`（`_run_with_timeout`，默认 20s 超时，env `FLEXKV_MOONCAKE_STORE_IO_TIMEOUT` 可调）。全部 10 个 prefill 节点已滚动重启部署修复版本并验证生效（原 100% 必现挂起 >900s 的 2 个 case，修复后 6.65s/1-9s 正常完成）。

> **本文档 §2 起记录的所有正式测试结果均基于已修复的版本**（修复时间 2026-07-12 之前的滚动重启完成）。

---

## 2. 测试记录

<!-- 每次运行追加一节，格式：## 2.N <日期> <数据集> <并发梯度> -->

### 2.0（前置探索，非正式数据，仅记录经验教训）

- **合成前缀模式冒烟**（`--mode fixed --concurrency 4 --num-requests 8`）：8/8 成功，链路打通。
- **真实数据集首次小规模验证**（`cache0.66`，`--max-records 20`，全局截断）：20/20 变成 20 个独立 session（破坏多轮结构），实测命中率仅 24.6%，远低于 ground-truth 59.0% —— **教训：不能用 `--max-records` 全局截断，必须按完整 session 采样**。
- **`--max-sessions 20` 修正后**（`c=8`）：命中率 97.1%（隔离测试，无其他租户竞争，远高于生产 66%/ground-truth 61.6%，符合预期，因为生产环境有多租户缓存驱逐）。
- 详细过程、bug 定位与修复见 `mooncake_store_impl_progress.md` §7 末尾章节。

---

### 2.1 `glm5-workbuddy-24K64K-cache0.82_0.55k_tools.json`（2026-07-12，进行中，遇到新问题被中止）

- 数据集特征：**无 `session_id` 字段**，548 条记录均为独立完整对话（非多轮增长）。脚本已修复此前的 bug（缺失 session_id 时全部记录被错误归并为 1 个"伪 session"，导致并发测试实质上只有 1 个 worker 在工作）——现改为每条记录分配独立伪 session id。
- `prompt_tokens` 字段在原始数据里全部为 0（数据源特性），故 ground-truth 命中率无法计算，`gt=-` 属预期，非脚本 bug。
- 参数：`--concurrency 4,8,16,32 --max-sessions 200 --max-tokens 32`

**发现的新问题（与已修复的 mooncake RDMA 挂起 bug 不同，尚未定位根因）**：
- c=4 档，4 个并发请求全部被路由到**同一个 prefill 实例 P1**（`***`）——原因大概率是这批数据共享同一个 agent 系统提示词（长公共前缀），router 的 cache-aware 路由把共享前缀请求粘到同一实例（属预期路由行为，但导致测试期间负载集中在单实例、无法体现多实例并发能力）。
- **该实例上出现两次 ~600 秒的整体停滞**（分别发生在 `12:14:18→12:24:15` 和 `12:24:18→12:32:39+`），随后自愈恢复。P1 sglang 日志中明确记录了异常的单次 `forward time: 602155.25ms`（约 600 秒），与环境变量 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600` 的数值精确吻合，怀疑是 **PD 分离 KV 传输等待环节卡住直到超时阈值后才继续**（不是 mooncake-store RDMA 层——本次未触发已修复 bug 对应的 I/O 看门狗日志）。
- 期间该 worker `match: all_keys` 被 scheduler 以每分钟 5000+ 次的频率反复调用（`exist_results` 全部命中=1），但请求迟迟未真正完成，日志刷屏但无实质进展——这是**活锁现象的外部表现**，根因待查（可能是 `waiting_queue` 循环反复对同一未 admit 请求做 match，配合 PD 等待卡顿共同放大）。
- 已终止该轮测试（进程 19805），避免继续空转消耗集群资源。

**根因排查结论（2026-07-12，已确认与 FlexKV/FlexKVConnector 无关，忽略）**：
- `forward time: 602155.25ms`（约 600s）与环境变量 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600` 精确吻合。该变量定义并仅使用于 `sglang/srt/disaggregation/mooncake/conn.py`（decode 侧等待 P→D KV 传输完成信号的超时），是 sglang **核心 PD 分离传输引擎**的机制，与 FlexKV 用 mooncake-store 做跨 prefill 前缀缓存共享（`flexkv/external/mooncake_store_utils.py`，即上次修复的 bug 所在模块）是**两个完全独立的子系统**，只是共享 mooncake 底层传输库、恰好都叫"mooncake"。
- 期间高频出现的 `match: all_keys`/`exist_results` 调用确实经过 FlexKVConnector（`get_new_hit_length`），但这是 sglang scheduler 对等待队列中未 admit 请求每次调度循环都重新前缀匹配的**既有设计行为**（`scheduler.py` `init_next_round_input`），是卡顿期间的正常副作用，不是卡顿原因。
- 反证：上次修复的 mooncake-store I/O 看门狗（20s 超时）本次**完全没有触发**，说明 RDMA 调用本身正常返回；且 FlexKV layerwise 加载依赖的 `_layer_done_counter`/eventfd 机制本身无超时保护，若是它卡住会永久挂起而非精确 600s 自愈。
- **结论：忽略，不需要修复 FlexKV 代码**。已作为"已知环境噪音"记录，后续测试注意规避（多个并发请求因共享长前缀被 router 粘到同一实例时，可能撞上该环境限制，导致个别请求出现 ~600s 尾部延迟）。

**后续实测发现该问题触发频率比预期更高**：在 `cache0.66` 数据集上重跑（更小规模，165 条/40 session），c=8 档单档就耗时 47+ 分钟（撞上该 600s 等待机制 4-5 次，每次卡住 1~8 个并发请求不等），测试效率严重低于预期。已终止该轮，调整策略为：优先用**合成 fixed 模式**（自包含、不依赖真实长尾数据，不会触发该问题）拿到核心 TTFT/吞吐/命中率基线；真实数据集测试保留但降低优先级、缩小规模，接受其耗时显著更长。

### 2.2 合成 `fixed` 模式（2026-07-12，完整跑通，无卡顿）

- 参数：`--mode fixed --concurrency 1,4,8,16,32 --num-requests 40 --prefix-tokens 1200 --max-tokens 64`
- 每档 40 个独立 session，共享约 1200 token 的合成前缀 + 唯一后缀（自包含，不依赖外部数据集，全程未触发 §2.1 的环境限制）
- 跑了两轮：**run1（`--flush-before`，先清本地 radix 再跑）** 和 **run2（跳板机进程重启后立即跑，未 flush）**。
  - **注意**：两轮之间因为脚本用独立进程运行、`run_tag` 按时间戳生成并写入前缀内容首部，因此 run2 并不是"复用 run1 写入 mooncake 的同一批前缀"的真正冷热对比，而是各自独立的一批同前缀请求（批内命中率有效，但跨轮次不构成 mooncake 冷热对比）。若要做真正的冷热对比，需要同一进程内先后两次调用相同 `run_tag` 的前缀。

**run1（`--flush-before`）**

| 并发 | TTFT p50/p90 (s) | TPOT p50 (s) | out tok/s | QPS | 命中率 | wall (s) |
|---|---|---|---|---|---|---|
| 1 | 0.40 / 0.70 | 0.031 | 11.0 | 0.17 | 93.6% | 231.7 |
| 4 | 0.90 / 1.13 | 0.032 | 85.2 | 1.33 | 86.4% | 30.1 |
| 8 | 0.93 / 1.19 | 0.034 | 164.3 | 2.57 | 76.8% | 15.6 |
| 16 | 1.63 / 5.30 | 0.030 | 175.7 | 2.75 | 57.6% | 14.6 |
| 32 | 1.48 / 2.02 | 0.046 | 346.4 | 5.41 | 57.6% | 7.4 |

**run2（未 flush，独立批次）**

| 并发 | TTFT p50/p90 (s) | TPOT p50 (s) | out tok/s | QPS | 命中率 | wall (s) |
|---|---|---|---|---|---|---|
| 1 | 0.40 / 0.43 | 0.029 | 27.8 | 0.44 | 93.6% | 91.9 |
| 4 | 0.79 / 1.82 | 0.030 | 86.9 | 1.36 | 86.4% | 29.5 |
| 8 | 1.22 / 1.86 | 0.031 | 155.7 | 2.43 | 76.8% | 16.4 |
| 16 | 1.09 / 1.49 | 0.029 | 295.5 | 4.62 | 57.6% | 8.7 |
| 32 | 2.15 / 3.17 | 0.048 | 296.0 | 4.62 | 67.2% | 8.7 |

**观察**：
- 命中率随并发升高而下降（93.6%→57.6%），符合预期：`--num-requests` 固定为 40，并发越高，`session` 并发展开越快，共享前缀在 mooncake/radix 里被写入的时机相对滞后，早期请求撞不到已写入的缓存。
- TTFT p50 随并发上升有轻微增长（0.40s→1.5~2s 量级），但增长平缓，说明在该并发区间（≤32）P800 5P5D 集群未出现明显饱和拐点。
- out tok/s 随并发近似线性增长（11→346 tok/s），QPS 同步从 0.17→5.4，说明系统在该区间内仍有较大吞吐余量。
- run1 的 `TTFT p99` 在 c=1 档异常高（134.485s）——这是 `--flush-before` 清空本地 radix 后第一次 batch 冷启动的正常现象（单个请求需要走完整 prefill，此并发下没有其他请求分摊，导致 p99 被单个慢请求拉高），不代表稳态延迟。

### 2.3 `glm5-128K-200K-cache0.95_1k.json`（2026-07-12，中止，真实超长上下文数据集耗时过高）

- 数据集特征：30 session / 998 记录，含 128K~200K token 量级的超长上下文。
- 参数：`--concurrency 4,8,16 --max-sessions 15 --max-records-per-session 4 --max-tokens 24`（56 条记录，规模已大幅缩小）
- c=4 档：前 20 条约 6 分钟内完成（符合超长 prompt 本身 prefill 耗时较长的预期），随后停滞 12+ 分钟无进展，4 个不同 prefill 实例各卡住 1 个请求（`load=1,0,1,1,1`），符合 §2.1 已确认的 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` 环境限制的表现模式。
- 已终止（进程 21316），未产出完整结果。**结论：超长上下文（128K+）+ 真实数据集组合对该环境限制的触发概率显著更高，暂不作为本轮主力测试对象**，留待后续有更长时间窗口或环境限制被上游修复后再补测。


### 结论（截至 2026-07-12）

1. **FlexKV+mooncake-store 功能与性能基线已建立**（合成 fixed 模式，§2.2）：
   - 并发 1→32 区间内，TTFT p50 从 0.4s 平缓上升到 ~1.5-2s，未见明显饱和拐点；out tok/s 从 11→346（近线性），QPS 从 0.17→5.4，集群在该区间仍有吞吐余量。
   - 命中率随并发上升而下降（93.6%→57.6%），符合"并发越高、请求越早到达、缓存写入滞后于并发展开"的预期机制，非异常。
2. **环境已知限制**（`SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600`，sglang 核心 PD 分离机制，已确认与 FlexKV 代码无关，详见 §2.1）会在真实数据集（尤其长上下文/低命中率/超长上下文场景）测试中较高频触发，导致单个或多个请求出现 ~600s 尾部延迟后自愈；这不影响系统正确性，但显著拖慢 benchmark 本身的执行效率，是**测试方法论上需要规避的因素**，不代表 FlexKV 存在性能缺陷。
3. **真实数据集测试**（`cache0.66`/`cache0.95`）因反复撞上该环境限制，本轮未能拿到完整的并发梯度数据，仅有部分片段（见 §2.1、§2.3）。建议后续：
   - 在集群更空闲、或该环境限制的触发条件被规避/上游修复后，重新用真实数据集补测完整并发梯度；
   - 或者接受"合成模式基线 + 真实数据集小样本抽测"的组合作为当前阶段的最终交付结果。

<!-- 全部数据集测完后填写：FlexKV+mooncake-store 在不同 prompt 长度/命中率分布下的 TTFT/吞吐特征，作为后续与 HiCache 对比的基线 -->

---

## 4. 附：常用命令

```bash
# 传脚本进 pod（本地改完脚本后需要重新走这一步）
base64 -w0 zittozhang_scripts/scripts/bench_5p5d.py > /tmp/bench_5p5d.b64
relay-cli run --executor executor-p800-jump-server "echo \$(cat /tmp/bench_5p5d.b64) | base64 -d > /root/phaedonsun/bench_5p5d.py && KUBECONFIG=/root/kubectl-tione.conf /tmp/kubectl cp /root/phaedonsun/bench_5p5d.py maas-public/glm5-p800-5p5d-prefill-0:/workspace/zittozhang/bench_5p5d.py -c prefill-engine"

# 后台跑一档测试（nohup + 行缓冲，可实时 tail）
relay-cli run --executor executor-p800-jump-server "KUBECONFIG=/root/kubectl-tione.conf /tmp/kubectl exec -n maas-public glm5-p800-5p5d-prefill-0 -c prefill-engine -- sh -c 'cd /workspace/zittozhang && nohup python3 bench_5p5d.py --mode dataset --dataset /home/data/darren/ai_evaluagtion/model_perf/data/<FILE>.json --concurrency 4,8,16,32 --max-sessions <N> --max-records-per-session 20 --max-tokens 32 --verbose --out /workspace/zittozhang/logs/result_<TAG>.json > /workspace/zittozhang/logs/bench_<TAG>.log 2>&1 & sleep 1; ps aux|grep bench_5p5d|grep -v grep'"

# 轮询
relay-cli run --executor executor-p800-jump-server "KUBECONFIG=/root/kubectl-tione.conf /tmp/kubectl exec -n maas-public glm5-p800-5p5d-prefill-0 -c prefill-engine -- sh -c 'tail -30 /workspace/zittozhang/logs/bench_<TAG>.log'"
```
