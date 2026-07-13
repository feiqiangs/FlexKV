# P800 FlexKV 专属 20 台节点 — 调度与占用记录

更新时间：2026-07-09 15:40

> 集群管理员在原 P800 集群里为 flexkv 腾出的一批专属机器。本文件记录调度所需的 `nodeSelector`、节点 IP 清单，以及最近一次占用/空闲巡检结果。IP 清单同步维护在仓库根目录 `p800_hosts.txt`。

---

## 1. nodeSelector（部署时必须原样照抄）

部署 Pod 时在 workload 的 `spec.template.spec.nodeSelector`（Deployment/StatefulSet）或 `spec.nodeSelector`（裸 Pod）里写入以下 5 条，Pod 才会被调度到这批专属节点：

```yaml
nodeSelector:
  ***: ***
  ***: null
  ***: ***
  ***: ***
  ***: null
```

### 逐条含义

| 键 | 值 | 语义 |
|---|---|---|
| `***` | `***` | 工作节点（区别于 master/控制面）|
| `***` | `***` | 昆仑芯 P800 机型 |
| `***` | `***` | 资源组 = flexkv 专属（核心专属标记，隔离其他业务）|
| `***`（网关征用标签）| `null` | 要求该标签**不存在**（排除被网关征用的节点）|
| `***`（健康状态标签）| `null` | 要求该标签**不存在**（排除被健康检查标异常的节点）|

> **`null` 语义**：标准 k8s `nodeSelector` 为等值匹配，`null` 非合法匹配值，此处是平台/模板约定，等价于"该节点上不存在此标签"。用原生 kubectl 验证时对应 `!***,!***`（两个"不存在"标签）。

### 标签匹配核验（2026-07-09）
- 用 5 条标签（含两个"不存在"）实测匹配到 **24 台**，全部 `Ready`。
- 这 24 台本来就都没有那两个"不存在"标签，故 `null` 条件不额外排除任何节点。
- **24 台 − 4 台旧 `glm5-p800-flexkv-inference`（`***`/`***`/`***`/`***`，仍 Running）= 本文件的 20 台。**
- ⟹ `p800_hosts.txt` 的 20 个 IP **全部命中** nodeSelector，无 MISS；nodeSelector 无需改动。

---

## 2. 20 台节点 IP 清单（= `p800_hosts.txt`）

```
***    ***    ***    ***    ***
***   ***    ***    ***    ***
***    ***    ***   ***   ***
***   ***   ***   ***   ***
```

---

## 3. 占用/空闲巡检（2026-07-09 15:40）

**结论：20 台全部空闲可用（无任何推理/业务 Pod）。**

- 每台仅运行"每节点都有"的基础 DaemonSet/系统组件，不计入业务占用：
  - `default/node-shell-*`（临时调试 pod）
  - `noaheepro/noaheepro-agent-*`（节点 agent，DaemonSet）
  - 少数节点另有 `monitoring-system/vmagent`、`local-path-storage/local-path-provisioner`、`default/p800-node-watcher`（均为基础设施组件）
- `maas-public` 命名空间在这 20 台上**均为空**——没有 sglang/mooncake/flexkv/glm5 等推理业务负载。
- 上一轮巡检中占用 `***` / `***` / `***` / `***` 的 `glm5-p800-mc-availability-test-0` **已释放**，这 4 台现已空闲。

| 空闲/占用 | 数量 |
|---|---|
| ✅ 空闲（无业务 Pod）| **20 台** |
| ⚠️ 业务占用 | 0 台 |

---

## 4. 备注
- 旧 `glm5-p800-flexkv-inference` A/B 实例（`***`/`***` = A prefill；`***`/`***` = B decode）仍在 Running，但**不属于**本批 20 台（不带入 `p800_hosts.txt`）。跨实例共享排查见 `doc/mooncake_store_impl_status.md` §7。
- 巡检脚本：`/tmp/match_flexkv_ips.sh`（IP↔标签比对）、`/tmp/check20_occupancy.sh`（逐台占用），经 relay-cli → 跳板机 `/tmp/kubectl` 执行。
