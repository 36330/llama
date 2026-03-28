# llama.cpp 调试与性能优化指南

> 本文档覆盖从算子级微观调试到系统级宏观优化的完整流程，基于 llama.cpp 源码分析。

---

## 目录

1. [算子级插桩 — 计算图 Per-Node Timing](#1-算子级插桩--计算图-per-node-timing)
2. [NVIDIA 工具链 — nsys + ncu 全流程](#2-nvidia-工具链--nsys--ncu-全流程)
3. [Pipeline 优化 — 计算访存 Overlap](#3-pipeline-优化--计算访存-overlap)
4. [宏观优化 — 计算图优化与系统级方案](#4-宏观优化--计算图优化与系统级方案)
5. [关键文件速查表](#5-关键文件速查表)

---

## 1. 算子级插桩 — 计算图 Per-Node Timing

### 1.1 llama.cpp 计算图执行流程

```
llama_decode() / llama_encode()
  └─> llama_context::decode()          // src/llama-context.cpp
       └─> process_ubatch()            // 构建 ggml_cgraph
            └─> ggml_backend_sched_graph_compute_async(sched, gf)  // :2119
                 └─> ggml_backend_sched_compute_splits(sched)      // ggml/src/ggml-backend.cpp:1445
                      └─> 遍历 splits[]
                           └─> ggml_backend_graph_compute_async(split_backend, &split->graph)  // :1586
                                └─> backend->iface.graph_compute(backend, cgraph)  // :364-367
                                     └─> [CUDA] ggml_backend_cuda_graph_compute()  // ggml-cuda.cu:4063
                                          └─> ggml_cuda_compute_forward() 对每个 node  // :2442
```

### 1.2 方式一：eval callback（零侵入，推荐入门）

llama.cpp 已有内置的 eval callback 机制，可以在每个 node 执行前后触发回调，无需修改源码。

**原理：** `ggml/include/ggml-backend.h:296-303`

```c
// 当 ask == true: 调度器询问你是否要观察这个 node
// 当 ask == false: 调度器把执行完的 node 传给你
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t, bool ask, void * user_data);
```

**注意：** 设置 callback 后，调度器会逐 node 执行而非批量执行（见 `ggml-backend.cpp:1590-1622`），这会引入额外开销。仅用于调试，不要在生产环境启用。

**在 llama-cli 中使用的方法：**

在 `src/llama-context.cpp:1136` 附近，callback 通过 `cparams.cb_eval` 传入：
```c
ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
```

**自己写一个 timing callback 的示例：**

```c
#include "ggml.h"
#include <stdio.h>
#include <map>
#include <string>

struct node_timing {
    int64_t start_us;
    std::map<std::string, double> total_us;  // op_name -> 累计时间
    std::map<std::string, int> count;        // op_name -> 调用次数
};

// 回调函数
bool timing_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * timing = (node_timing *)user_data;

    if (ask) {
        // 记录开始时间，返回 true 表示"我要观察这个 node"
        timing->start_us = ggml_time_us();
        return true;
    }

    // ask == false: node 已执行完毕
    int64_t elapsed = ggml_time_us() - timing->start_us;
    std::string name = ggml_op_name(t->op);

    timing->total_us[name] += elapsed;
    timing->count[name]++;

    return true;  // 返回 false 会取消整个图的计算
}

// 使用方式：在创建 llama_context 前设置
// params.cb_eval = timing_callback;
// params.cb_eval_user_data = &my_timing_data;
```

**断点位置（GDB）：**

```bash
# 在每个 node 执行前断住
b ggml-backend.cpp:1596   # callback_eval(t, true, ...) 即 ask 阶段
b ggml-backend.cpp:1616   # callback_eval(t, false, ...) 即观察阶段

# 查看当前 node 信息
p t->op
p ggml_op_name(t->op)
p t->name
p t->ne[0], t->ne[1], t->ne[2], t->ne[3]
```

### 1.3 方式二：修改 ggml_backend_sched_compute_splits()（更精确）

直接在调度器的 split 执行循环中插桩。适合你想看到每个 split（子图）的整体耗时。

**插桩位置：** `ggml/src/ggml-backend.cpp:1585`

```c
// 在现有代码前后加计时
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    // ... 输入拷贝逻辑 ...

    int64_t t0 = ggml_time_us();  // <-- 插入

    if (!sched->callback_eval) {
        enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
        // ...
    }

    ggml_backend_synchronize(split_backend);  // 确保 GPU 完成
    int64_t t1 = ggml_time_us();  // <-- 插入
    printf("split %d (backend %d): %d nodes, %.2f ms\n",
           split_id, split_backend_id, split->graph.n_nodes,
           (t1 - t0) / 1000.0);
}
```

**注意：** 必须在计时前调用 `ggml_backend_synchronize()`，否则 GPU 异步执行会导致时间不准。

### 1.4 方式三：CUDA 后端 per-kernel timing（最精确）

在 CUDA 层面用 cudaEvent 精确测量每个算子的 GPU 执行时间。

**插桩位置：** `ggml/src/ggml-cuda/ggml-cuda.cu:2442`

```c
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // --- 插入开始 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, ctx.stream());
    // --- 插入结束 ---

    // 原有的 switch(dst->op) 逻辑 ...
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        // ...
    }

    // --- 插入开始 ---
    cudaEventRecord(stop, ctx.stream());
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[CUDA] op=%s name=%s %.3f ms\n", ggml_op_name(dst->op), dst->name, ms);
    fflush(stdout);  // 加这行
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // --- 插入结束 ---

    return true;
}
```

**注意：** `cudaEventSynchronize` 会强制同步，会严重影响性能。仅在分析时使用。如果不想同步，可以把 event 对存起来，在图执行完后统一读取。

### 1.5 已有的调试工具

llama.cpp 自带一些可视化和调试功能：

```c
// 打印图的统计信息（节点数、叶子数等）
ggml_graph_print(cgraph);    // ggml/include/ggml.h:2652

// 导出 .dot 格式的图，可用 graphviz 可视化
ggml_graph_dump_dot(cgraph, NULL, "graph.dot");  // :2655

// 然后用 graphviz 渲染
// dot -Tpng graph.dot -o graph.png
```

---

## 2. NVIDIA 工具链 — nsys + ncu 全流程

### 2.1 编译准备

profiling 用 `RelWithDebInfo` 模式编译，既有优化又保留调试符号：

```bash
cd ~/llama/llama.cpp
rm -rf build && mkdir build && cd build

# RelWithDebInfo: -O2 + 调试符号
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=ON
cmake --build . -j$(nproc)

# 如果要 CUDA 行级 profiling，加 -lineinfo
# cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#   -DGGML_CUDA=ON \
#   -DCMAKE_CUDA_FLAGS="-lineinfo"
```

### 2.2 nsys — Nsight Systems（系统级 Profiling）

nsys 用于宏观分析：CPU 和 GPU 的时间线、kernel 占比、内存拷贝、空闲时间。

**基础用法：**

```bash
# 基本 profile
nsys profile -o llama_profile \
  ./bin/llama-cli -m /path/to/model.gguf -ngl 100 -p "Hello" -n 32

# 生成 .nsys-rep 文件，用 Nsight Systems GUI 打开
```

**常用参数：**

```bash
nsys profile \
  -o output_name \             # 输出文件名
  --trace=cuda,nvtx,osrt \     # 追踪 CUDA API + NVTX 标记 + OS runtime
  --cuda-memory-usage=true \   # 追踪显存分配
  --gpu-metrics-device=all \   # GPU 硬件计数器
  --force-overwrite=true \     # 覆盖已有文件
  --stats=true \               # 运行后打印统计摘要
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 64
```

**命令行快速分析（不用 GUI）：**

```bash
# 导出统计信息
nsys stats llama_profile.nsys-rep

# 查看 CUDA kernel 统计
nsys stats --report cuda_gpu_kern_sum llama_profile.nsys-rep

# 查看 CUDA API 调用统计
nsys stats --report cuda_api_sum llama_profile.nsys-rep

# 查看内存拷贝统计
nsys stats --report cuda_gpu_mem_size_sum llama_profile.nsys-rep
```

**nsys 看什么：**

| 关注点 | 怎么看 |
|--------|--------|
| 哪个 kernel 最耗时 | `cuda_gpu_kern_sum` 报告，按 Total Time 排序 |
| CPU 是否成为瓶颈 | timeline 中看 CPU 和 GPU 是否有大段不重叠的时间 |
| 内存拷贝开销 | `cuda_gpu_mem_size_sum` 看 H2D/D2H 传输量和时间 |
| GPU 利用率 | timeline 中 GPU 行是否有大段空白 |
| kernel launch 开销 | 小 kernel 频繁 launch 会导致 CPU-bound |

### 2.3 ncu — Nsight Compute（Kernel 级深度分析）

ncu 用于分析单个 kernel 的性能瓶颈：占用率、内存带宽、计算吞吐。

**基础用法：**

```bash
# 分析所有 kernel（会非常慢，建议限制范围）
ncu --set full -o llama_ncu \
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 8

# 生成 .ncu-rep 文件，用 Nsight Compute GUI 打开
```

**精确分析特定 kernel：**

```bash
# 只分析包含 "flash_attn" 的 kernel
ncu --set full \
  --kernel-name "flash_attn" \
  -o attn_analysis \
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 8

# 只分析矩阵乘法相关 kernel
ncu --set full \
  --kernel-name "mul_mat" \
  -o matmul_analysis \
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 8

# 限制分析的 kernel 启动次数（避免太慢）
ncu --set full \
  --launch-skip 10 \      # 跳过前 10 次 launch
  --launch-count 5 \      # 只分析 5 次 launch
  -o limited_analysis \
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 8
```

**命令行快速查看：**

```bash
# 打印所有 kernel 的摘要
ncu --print-summary per-kernel \
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 8

# 查看特定指标
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
  ./bin/llama-cli -m model.gguf -ngl 100 -p "Hello" -n 8
```

**ncu 看什么 — 典型瓶颈判断：**

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| Compute (SM) Throughput 高 | 计算密集型 | 减少计算量、用更高效算法 |
| Memory Throughput 高 | 访存密集型 | 减少访存、提升缓存命中率 |
| 两者都低 | latency bound | 检查 occupancy、warp stall 原因 |
| Occupancy 低 | 并行度不够 | 调整 block size、减少寄存器/共享内存使用 |
| L2 Hit Rate 低 | 缓存未命中多 | 优化访存模式、数据布局 |

### 2.4 实战流程：从发现到优化

```
Step 1: llama-bench 建立 baseline
  └─> ./bin/llama-bench -m model.gguf -ngl 100
       记录 pp (prefill) 和 tg (token gen) 的 t/s

Step 2: nsys 系统级分析
  └─> 找到最耗时的 kernel（通常是 mul_mat 和 flash_attn）
  └─> 观察 CPU-GPU overlap、内存拷贝占比

Step 3: ncu 深入分析热点 kernel
  └─> 判断是 compute bound 还是 memory bound
  └─> 分析 occupancy、warp stall

Step 4: 针对性优化
  └─> memory bound → 考虑量化、减少数据搬运、优化访存模式
  └─> compute bound → 考虑 Tensor Core (WMMA/MMA)、算法优化
  └─> latency bound → 提升 occupancy、kernel fusion

Step 5: 验证
  └─> 重新跑 llama-bench，对比 baseline
```

---

## 3. Pipeline 优化 — 计算访存 Overlap

### 3.1 llama.cpp 现有的 Pipeline Parallelism

llama.cpp 已有 pipeline parallelism 机制，但它的目标是**多 GPU 间的流水线**，而非单 GPU 内的计算/访存 overlap。

**启用条件：** `src/llama-context.cpp:312-339`

```c
bool pipeline_parallel =
    model.n_devices() > 1 &&                     // 多 GPU
    model.n_gpu_layers() > model.hparams.n_layer && // 所有层都 offload
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER && // 按层分割
    cparams.offload_kqv &&                        // KQV offload
    !model.has_tensor_overrides();                // 无张量覆盖
```

**核心数据结构：** `ggml/src/ggml-backend.cpp:714-718`

```c
// pipeline parallelism support
int n_copies;     // 数据副本数量（用于 double/triple buffering）
int cur_copy;     // 当前使用的副本
int next_copy;    // 下一次使用的副本
ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
```

**工作原理：**

1. 图被按后端分割成多个 split（`ggml_backend_sched_split_graph`）
2. 每个 split 在对应后端上执行
3. split 之间的数据通过 `ggml_backend_tensor_copy_async()` 异步拷贝
4. 用 event 做同步：`event_record` 标记完成，`event_wait` 等待依赖

### 3.2 单 GPU 内的计算/访存 Overlap

llama.cpp 的 CUDA 后端已预留了多 stream 支持：

```c
// ggml/src/ggml-cuda/common.cuh:151
#define GGML_CUDA_MAX_STREAMS 8

// 每个 device 有 8 个 stream
cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];
```

**实现单 GPU 计算/访存 overlap 的方案：**

#### 方案 A：双 Stream Ping-Pong

目标：在 stream 0 上做计算的同时，在 stream 1 上做下一个 split 的数据拷贝。

```
时间线:
Stream 0: [compute split_0] ──────── [compute split_1] ────────
Stream 1:          [copy inputs for split_1] [copy inputs for split_2]
                   ↑                         ↑
                   event_wait                event_wait
```

**修改点在** `ggml_backend_sched_compute_splits()`（`ggml-backend.cpp:1445`）：

```c
// 伪代码示意
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    int compute_stream = split_id % 2;       // 交替使用 stream
    int copy_stream    = (split_id + 1) % 2; // 拷贝用另一个 stream

    // 在 copy_stream 上异步拷贝下一个 split 的输入
    if (split_id + 1 < sched->n_splits) {
        prefetch_inputs_async(splits[split_id + 1], copy_stream);
    }

    // 在 compute_stream 上等待拷贝完成（如果有依赖）
    if (split_id > 0) {
        cudaStreamWaitEvent(compute_stream, copy_done_event[split_id]);
    }

    // 在 compute_stream 上执行计算
    ggml_backend_graph_compute_async(backend, &split->graph);

    // 记录计算完成事件
    cudaEventRecord(compute_done_event[split_id], compute_stream);
}
```

#### 方案 B：利用 CUDA Graph 自动 Overlap

CUDA 后端已有 CUDA Graph 支持（`ggml-cuda.cu:4068-4117`，需 `USE_CUDA_GRAPH` 编译开关）。

CUDA Graph 可以让 runtime 自动发现并行机会：

```bash
# 编译时启用 CUDA Graph
cmake .. -DGGML_CUDA=ON -DUSE_CUDA_GRAPH=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

CUDA Graph 的工作流程：
1. 第一次执行：warmup，正常执行 kernel（`ggml-cuda.cu:4083-4091`）
2. 第二次执行且参数不变：capture 整个图（`cudaStreamBeginCapture`，`:4114`）
3. 后续执行：replay captured graph（跳过 CPU launch 开销）

#### 方案 C：MoE 权重的选择性拷贝（已实现）

llama.cpp 已经对 MoE (Mixture of Experts) 模型做了优化：只拷贝被选中的 expert 权重，而非整个权重矩阵。

见 `ggml-backend.cpp:1480-1564`：

```c
// 当检测到 MUL_MAT_ID 算子 + host 上的权重时
// 只拷贝 used experts，跳过未选中的 expert
if (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID) {
    // 读取 ids tensor，找出哪些 expert 被使用
    // 然后按连续区间批量拷贝
}
```

### 3.3 实际操作建议

| 场景 | 建议方案 |
|------|----------|
| 多 GPU 推理 | 直接使用已有的 pipeline parallelism（自动启用） |
| 单 GPU 大模型 | 尝试 CUDA Graph (`USE_CUDA_GRAPH=ON`)，减少 launch 开销 |
| MoE 模型 offload | 已自动优化，确保权重放在 host、计算在 GPU |
| 自定义 overlap | 修改 `ggml_backend_sched_compute_splits()`，用多 stream |

---

## 4. 宏观优化 — 计算图优化与系统级方案

### 4.1 计算图优化

#### 4.1.1 图分割策略

**现有机制：** `ggml_backend_sched_split_graph()`（`ggml-backend.cpp:925`）

调度器根据每个 node 的后端分配，自动将图分割成若干 split。每个 split 内的所有 node 在同一个后端上执行。

**优化方向：**

- **减少 split 数量：** 每次 split 切换都有数据拷贝和同步开销。可以通过调整 `ggml_backend_sched_set_tensor_backend()` 手动控制 tensor 放置，减少跨后端切换。

- **检查当前 split 数量：**
  ```c
  int n_splits = ggml_backend_sched_get_n_splits(sched);
  printf("graph split into %d parts\n", n_splits);
  ```

#### 4.1.2 算子融合（Op Fusion）

llama.cpp 的 ggml 层面目前**不做自动算子融合**。但在 CUDA 后端，某些复合操作已被手动融合：

- **Flash Attention：** 将 QKV matmul + softmax + attention 融合为一个 kernel
  - 多种实现：`fattn-mma-f16.cuh`（Tensor Core MMA）、`fattn-wmma-f16.cu`（WMMA）、`fattn-tile.cu`（分块）、`fattn-vec.cuh`（向量化）
  - 启用：`-fa` 参数（Flash Attention）

- **RoPE：** 旋转位置编码作为独立 kernel（`rope.cu`）

- **Fused Softmax：** 独立的 fused softmax kernel（`softmax.cu`）

**如果你要做新的融合：**

1. 在 `ggml.h` 中定义新的 `GGML_OP_XXX`
2. 在图构建阶段（`src/llama-graph.cpp`）使用新算子
3. 在 CUDA 后端（`ggml-cuda/`）实现对应 kernel
4. 在 `ggml_cuda_compute_forward()` 的 switch 中注册

#### 4.1.3 图可视化

```c
// 在 llama-context.cpp 中需要查看图结构时：
ggml_graph_dump_dot(gf, NULL, "/tmp/llama_graph.dot");

// 然后：
// dot -Tsvg /tmp/llama_graph.dot -o /tmp/llama_graph.svg
// 在浏览器中打开 svg 查看完整计算图
```

### 4.2 PD 分离（Prefill-Decode Separation）

#### 4.2.1 背景

- **Prefill（预填充）：** 处理整个 prompt，batch size 大，计算密集（compute bound）
- **Decode（解码）：** 逐 token 生成，batch size = 1，访存密集（memory bound）

这两个阶段的硬件利用特征完全不同，混合调度会导致资源浪费。

#### 4.2.2 llama.cpp 的现状

llama.cpp **没有显式的 PD 分离**。`llama_decode()` 通过 batch 大小隐式区分：
- prefill: `batch.n_tokens > 1`
- decode: `batch.n_tokens == 1`（或少量 token）

batch 分割逻辑在 `src/llama-batch.h`：

```c
class llama_batch_allocr {
    llama_ubatch split_simple(uint32_t n_ubatch);   // 简单按大小分割
    llama_ubatch split_equal(uint32_t n_ubatch);    // 等长分割
    llama_ubatch split_seq(uint32_t n_ubatch);      // 按序列分割
};
```

#### 4.2.3 实现 PD 分离的方案

**方案一：应用层分离（最简单）**

在调用 `llama_decode` 时自己控制：

```c
// Prefill 阶段：大 batch
llama_batch batch_prefill = llama_batch_init(prompt_len, 0, 1);
for (int i = 0; i < prompt_len; i++) {
    llama_batch_add(&batch_prefill, tokens[i], i, {0}, false);
}
llama_batch_get_one_last_token(&batch_prefill);  // 只需最后一个 token 的输出
llama_decode(ctx, batch_prefill);

// Decode 阶段：单 token
for (int i = 0; i < n_gen; i++) {
    llama_batch batch_decode = llama_batch_get_one(&next_token, 1);
    llama_decode(ctx, batch_decode);
    // sample next token...
}
```

**方案二：Server 层 Continuous Batching**

`llama-server` 已支持多请求并发处理。可以在 server 层实现：

- 优先处理新请求的 prefill（利用大 batch 的计算效率）
- 将已完成 prefill 的请求加入 decode batch
- 动态调整 batch 组合

关键参数：
```bash
./bin/llama-server \
  -m model.gguf \
  -ngl 100 \
  -c 16384 \          # 总 context 大小
  --parallel 4 \      # 并发 slot 数
  -cb                  # 启用 continuous batching
```

**方案三：双实例分离（激进方案）**

如果有多张 GPU，可以运行两个独立实例：
- GPU 0：专门跑 prefill（用大 batch，配置高 n_ubatch）
- GPU 1：专门跑 decode（用小 batch，配置低 n_ubatch）
- 中间通过 KV cache 迁移或 RPC 后端传递状态

### 4.3 Tensor Parallelism（张量并行）

**现有支持：** llama.cpp 支持跨多 GPU 的张量并行。

**启用方式：**

```bash
# 自动均匀分割
./bin/llama-cli -m model.gguf -ngl 100 --split-mode row

# 手动指定每个 GPU 的比例
./bin/llama-cli -m model.gguf -ngl 100 --split-mode row --tensor-split 0.5,0.5
```

**实现原理：** `ggml-cuda.cu` 中的 split buffer

```c
// 按行分割矩阵到多个 GPU
get_row_split(&row_low, &row_high, tensor, tensor_split, device_id);
```

每个 GPU 持有矩阵的一部分行，matmul 后需要 all-reduce 汇总结果。

### 4.4 KV Cache 优化

**关键文件：** `src/llama-kv-cache.h/cpp`

```bash
# 使用量化的 KV cache 减少显存
./bin/llama-cli -m model.gguf -ngl 100 -ctk q8_0 -ctv q8_0 -fa

# -ctk: K cache 的量化类型
# -ctv: V cache 的量化类型（需要 -fa Flash Attention）
```

**KV cache slot 管理：**

```c
// 查找可用 slot
slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

// 序列操作
bool seq_rm(seq_id, p0, p1);  // 删除序列的 [p0, p1) 范围
void seq_cp(src, dst, p0, p1); // 复制序列
void seq_add(seq_id, delta);   // 位置偏移（用于 KV cache shift）
```

---

## 5. 关键文件速查表

### 核心架构

| 文件 | 内容 |
|------|------|
| `ggml/include/ggml.h` | 张量、算子、计算图定义 |
| `ggml/include/ggml-backend.h` | 后端接口、调度器 API、eval callback |
| `ggml/src/ggml-backend.cpp` | 调度器实现、图分割、split 执行 |
| `ggml/src/ggml-backend-impl.h` | 后端实现者需要实现的接口 |
| `ggml/src/ggml.c` | ggml_time_us/ms 计时函数 |

### CUDA 后端

| 文件 | 内容 |
|------|------|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA 后端主文件、算子 dispatch、CUDA Graph |
| `ggml/src/ggml-cuda/common.cuh` | CUDA 上下文、stream/pool 管理 |
| `ggml/src/ggml-cuda/fattn*.cu/cuh` | Flash Attention 各种实现 |
| `ggml/src/ggml-cuda/mmq.cu/cuh` | 量化矩阵乘法 |
| `ggml/src/ggml-cuda/mmvq.cu/cuh` | 量化矩阵-向量乘法 |
| `ggml/src/ggml-cuda/softmax.cu` | Softmax kernel |
| `ggml/src/ggml-cuda/rope.cu` | RoPE 旋转位置编码 |
| `ggml/src/ggml-cuda/norm.cu` | LayerNorm / RMSNorm |

### llama 推理层

| 文件 | 内容 |
|------|------|
| `include/llama.h` | 公开 API（llama_decode, llama_encode） |
| `src/llama-context.cpp` | 推理主流程、pipeline parallel 判断 |
| `src/llama-graph.h/cpp` | 计算图构建、图输入管理 |
| `src/llama-model.cpp` | 模型加载、tensor offload |
| `src/llama-kv-cache.h/cpp` | KV cache 管理 |
| `src/llama-batch.h/cpp` | Batch 分割策略 |

### 调试相关断点速查

| 断点位置 | 用途 |
|----------|------|
| `ggml-backend.cpp:1586` | 每个 split 的图执行入口 |
| `ggml-backend.cpp:1596` | eval callback ask 阶段 |
| `ggml-backend.cpp:1616` | eval callback 观察阶段 |
| `ggml-cuda.cu:2442` | CUDA 算子 dispatch 入口 |
| `ggml-cuda.cu:4063` | CUDA 图级执行入口 |
| `llama-context.cpp:2119` | llama 调度器执行入口 |
| `llama-context.cpp:312` | pipeline parallel 判断逻辑 |
