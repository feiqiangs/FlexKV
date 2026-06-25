# FlexKV 适配 P800（昆仑芯）改造点清单

> 背景：P800 是类 CUDA 生态的 GPU，运行时通过 `xpurt` / `XMLIRRuntime` 提供 CUDA Runtime API 兼容 shim，但**不支持 NVIDIA 编译器后端**（nvcc / PTX / sm_XX）。本文档列出 FlexKV 当前代码中需要改造的所有点，按风险/优先级排序，作为 P800 适配工作的指引。

---

## 1. 改造点总览

| # | 改造点 | 风险等级 | 必须改 | 涉及文件 |
|---|---|---|---|---|
| 1 | PTX 内联汇编 kernel | 🔴 高 | ✅ | `csrc/transfer.cu` |
| 2 | `__global__` + `<<<>>>` 启动语法 | 🔴 高 | ✅ | `csrc/transfer.cu`、`csrc/gds/layout_transform.cu` |
| 3 | `.cu` 文件 + nvcc 编译流水线 | 🔴 高 | ✅ | `setup.py`、`build.sh` |
| 4 | NVTX profiling 头文件 | 🟡 中 | ✅ | `csrc/bindings.cpp`、`csrc/layerwise.h/cpp` |
| 5 | GDS（cufile）依赖 | 🟡 中 | ✅ | `setup.py`、`csrc/gds/*` |
| 6 | `TORCH_CUDA_ARCH_LIST` 自动检测 | 🟡 中 | ✅ | `setup.py:detect_cuda_arch` |
| 7 | CUDA Runtime API（兼容层覆盖） | 🟢 低 | ❌ | 多处 |

---

## 2. 必须改造的点（详细）

### 2.1 🔴 PTX 内联汇编 kernel —— 必须替换

**位置**：`FlexKV/csrc/transfer.cu`（约 L67–L75）

**问题代码**：

```cpp
asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];"
             : "=f"(element.x), "=f"(element.y), "=f"(element.z), "=f"(element.w)
             : "l"(&FLOAT4_PTR(src_chunk_ptr)[idx])
             : "memory");
asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};"
             ::"l"(&FLOAT4_PTR(dst_chunk_ptr)[idx]),
               "f"(element.x), "f"(element.y), "f"(element.z), "f"(element.w)
             : "memory");
```

**为什么不行**：`ld.global.nc.v4.f32` / `st.global.cg.v4.f32` 是 NVIDIA 专属 PTX 指令（sm_XX 架构），P800 的 XPU 编译器无法识别。

**改造方案（任选一种）**：

- **方案 A（推荐，工作量最小）**：删除自写 kernel 模式，全部走 CE 模式（即 `cudaMemcpyAsync`）。CUDA Runtime shim 在 P800 上可用，功能正确，性能可接受。
- **方案 B**：替换为 P800 原生 H2D/D2H 接口：

  ```cpp
  RUN_XPU(baidu::xpu::api::do_host2device, ctx, src_ptr, dst_ptr, bytes);
  RUN_XPU(baidu::xpu::api::do_device2host, ctx, src_ptr, dst_ptr, bytes);
  ```

- **方案 C**：用 XHPC 算子库（`xdnn::*`）实现等价的 strided gather/scatter copy。

---

### 2.2 🔴 `__global__` kernel + `<<<>>>` 启动语法 —— 必须替换

**位置**：

- `FlexKV/csrc/transfer.cu` —— `transfer_kv_blocks_kernel`
- `FlexKV/csrc/gds/layout_transform.cu` —— `layout_transform_kernel`

**问题代码**：

```cpp
__global__ void transfer_kv_blocks_kernel(...) { ... }

transfer_kv_blocks_kernel<Type><<<gridDim, blockDim, 0, stream>>>(...);
```

**为什么不行**：`__global__` / `__device__` 标识、`<<<...>>>` 启动语法只能由 nvcc 解析，g++ 无法编译；P800 没有 nvcc。

**改造方案**：参考 P800 sglang 的写法，把 kernel 替换为 XPU API 调用：

```cpp
auto ctx = get_xmlir_context_with_stream();
RUN_XPU(baidu::xpu::api::xxx_op, ctx, args...);
```

如果暂时不想引入 XPU API，可以**直接删掉自写 kernel 模式**，强制 `use_ce_transfer=true`，全部走 `cudaMemcpyAsync`。

---

### 2.3 🔴 `.cu` 文件 + nvcc 编译流水线 —— 必须改造

**位置**：`FlexKV/setup.py`

**问题代码**：

```python
cpp_sources = [
    "csrc/bindings.cpp",
    "csrc/transfer.cu",          # ← 走 nvcc
    ...
]

extra_compile_args = {"nvcc": nvcc_compile_args, "cxx": extra_compile_args}
cpp_extension.CUDAExtension(...)
```

**为什么不行**：P800 工具链没有 `nvcc`，`CUDAExtension` 找不到 nvcc 会编译失败。

**改造方案**：参考 P800 sglang `setup_klx.py`：

1. 把 `transfer.cu` → `transfer.cc`（同时移除 PTX / `__global__`）。
2. 把 `gds/layout_transform.cu` → `.cc`。
3. `extra_compile_args` 移除 `"nvcc"` 段，只保留 `"cxx"`。
4. `setup.py` 中 `libraries` 仍可保留 `cudart`、`c10_cuda`（P800 提供 shim），但需要新增 `xpurt`、`xpuapi`、`XMLIRRuntime`、`bkcl` 等。

---

### 2.4 🟡 NVTX profiling 头文件 —— 必须改造

**位置**：

- `FlexKV/csrc/bindings.cpp:11`
- `FlexKV/csrc/layerwise.h:12`
- `FlexKV/csrc/layerwise.cpp:8`

**问题代码**：

```cpp
#include <nvtx3/nvToolsExt.h>
nvtxRangePushA("transfer_h2d");
nvtxRangePop();
```

**为什么不行**：NVTX 是 NVIDIA Nsight 工具链专属，P800 工具链没有这个头文件，编译期直接报 `No such file or directory`。

**改造方案**：用宏开关包起来，P800 上空操作：

```cpp
#ifdef FLEXKV_ENABLE_NVTX
  #include <nvtx3/nvToolsExt.h>
  #define FLEXKV_NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
  #define FLEXKV_NVTX_RANGE_POP()      nvtxRangePop()
  #define FLEXKV_NVTX_MARK(name)       nvtxMarkA(name)
#else
  #define FLEXKV_NVTX_RANGE_PUSH(name) ((void)0)
  #define FLEXKV_NVTX_RANGE_POP()      ((void)0)
  #define FLEXKV_NVTX_MARK(name)       ((void)0)
#endif
```

`setup.py` 中 P800 默认**不定义** `FLEXKV_ENABLE_NVTX`。后续若需要 P800 profiling，可桥接到百度 XHPC 的 profiler API。

---

### 2.5 🟡 GDS（cufile）依赖 —— 必须关闭

**位置**：

- `FlexKV/setup.py`：`extra_link_args.append("-lcufile")`
- `FlexKV/csrc/gds/*`

**问题**：`cufile` 是 NVIDIA GPUDirect Storage 库，P800 上不存在。

**改造方案**：

1. 编译时强制 `FLEXKV_ENABLE_GDS=0`，跳过 GDS 源码编译。
2. 或者在 `setup.py` 中检测目标平台：若是 P800，自动忽略 GDS 选项。
3. 上层代码（`flexkv/transfer/*`）中 GDS 相关分支需要确保在 `enable_gds=False` 时不走到。

---

### 2.6 🟡 `TORCH_CUDA_ARCH_LIST` 自动检测 —— 必须改造

**位置**：`FlexKV/setup.py:detect_cuda_arch`

**问题代码**：

```python
def detect_cuda_arch():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            archs.add(f"{major}.{minor}")
    ...
    fallback = "8.0;8.6;9.0"
```

**问题**：

- P800 上 `torch.cuda.get_device_capability` 返回值不一定合理。
- 即使返回了值，nvcc 也用不上（因为已经不走 nvcc 了）。

**改造方案**：

- 既然已经移除 nvcc 路径（见 2.3），可以**完全去掉** `detect_cuda_arch` 调用和 `TORCH_CUDA_ARCH_LIST` 设置。
- 或加平台开关：仅在标准 NVIDIA 环境下设置。

---

## 3. 可以保留不改的点

P800 的 `xpurt` 提供了 CUDA Runtime API 的兼容 shim，以下调用 **API 名字不变即可正常运行**：

| API | FlexKV 中的位置 |
|---|---|
| `cudaMemcpyAsync` / `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost` | `csrc/transfer.cu` (CE 模式) |
| `cudaHostRegister` / `cudaHostUnregister` | `flexkv/transfer/host_buffer.py`、`flexkv/transfer/worker.py` |
| `cudaMallocHost` / `cudaFreeHost` | `csrc/tp_transfer_thread_group.cpp` |
| `cudaSetDevice` / `cudaStreamCreate` / `cudaStreamSynchronize` | `csrc/tp_transfer_thread_group.cpp` |
| `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` | （FlexKV 当前少用，sglang 用） |
| `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` | `flexkv/common/memory_handle.py` |
| `cudaError_t` / `cudaGetErrorString` | 各处错误处理 |
| `#include <cuda_runtime.h>` | 各 `.cu`/`.cpp` 头部 |
| 链接 `-lcudart` | `setup.py` |

> ⚠️ **注意**：`cudaIpcMemHandle_t` 在 P800 上的二进制大小要确认是否仍是 64 字节，否则 `flexkv/common/memory_handle.py` 中的 ctypes 结构体大小校验需要调整。

---

## 4. 改造优先级与执行顺序建议

### 阶段 1：让 FlexKV 能在 P800 编译通过（必做）

1. 关闭 NVTX（2.4）—— 加宏开关，构建参数中不定义 `FLEXKV_ENABLE_NVTX`
2. 关闭 GDS（2.5）—— 设 `FLEXKV_ENABLE_GDS=0`
3. 移除 nvcc 路径（2.3 + 2.6）—— `setup.py` 改造，`.cu` → `.cc`，去掉 arch 检测

### 阶段 2：让 FlexKV 能在 P800 运行（必做）

4. 替换 PTX kernel（2.1）—— 推荐方案 A：删掉自写 kernel 模式，强制 CE 模式
5. 替换 `__global__` kernel（2.2）—— 一并完成
6. 验证 CUDA Runtime shim 可用（3）—— 跑 H2D/D2H 单测

### 阶段 3：性能优化（可选）

7. 用 `baidu::xpu::api::do_host2device/do_device2host` 替代 `cudaMemcpyAsync`，对比性能
8. 用 XHPC 算子库实现高效 layout transform（替代 `csrc/gds/layout_transform.cu`）
9. 桥接 NVTX 到 XHPC profiler

---

## 5. 关键参考

- P800 sglang 编译方式：`/data1/home/phaedonsun/p800/sglang/sgl-kernel/setup_klx.py`
- P800 sglang H2D 原生写法：`/data1/home/phaedonsun/p800/sglang/sgl-kernel/csrc/klx/klx_host2device.cc`
- P800 sglang CUDA shim 用法（cudaMemcpyAsync）：`/data1/home/phaedonsun/p800/sglang/sgl-kernel/csrc/kvcacheio/transfer.cc`
- P800 sglang XHPC 算子调用范式：`/data1/home/phaedonsun/p800/sglang/sgl-kernel/csrc/klx/reshape_and_cache.cc`

---

## 6. 一句话总结

> P800 通过 **xpurt** 兼容了 NVIDIA CUDA Runtime API（`cudaMemcpy*` / `cudaHostRegister*` / `cudaIpc*` 等可直接复用），但**不兼容 NVIDIA 编译器后端**——所有 `.cu` 文件、`__global__` kernel、`<<<>>>` 启动、PTX 内联汇编、cufile、NVTX 头文件都必须替换或裁掉，改用 `baidu::xpu::api::*`（XDNN/XHPC）算子库 + g++ 编译流水线。
