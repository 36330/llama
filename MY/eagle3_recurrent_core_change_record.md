# EAGLE3 Recurrent Core Change Record

更新时间：基于当前工作区代码现状整理

## 1. 这次改动的目标

这轮改动的核心目标是：

- 保留 `speculative-simple` 原来的“target 一次 batch verify”形态
- 不再使用 example 层面的串行逐 depth `llama_decode()`
- 为 hybrid/recurrent 架构补上 EAGLE3 一轮验证期间的中间 recurrent state 保存能力
- 在 accept 后，直接把“接受到的那个 depth 对应 recurrent state”提升为 live state
- 尽量不改原有 KV cache 语义，KV 仍复用原有 `seq_rm()` 清理逻辑

---

## 2. 这次改动过的文件

本轮涉及 10 个代码文件：

- `examples/speculative-simple/speculative-simple.cpp`
- `include/llama.h`
- `src/llama-context.cpp`
- `src/llama-memory.h`
- `src/llama-memory-hybrid.h`
- `src/llama-memory-hybrid.cpp`
- `src/llama-memory-recurrent.h`
- `src/llama-memory-recurrent.cpp`
- `src/models/qwen35.cpp`
- `src/models/qwen35moe.cpp`

---

## 3. 按文件说明改了什么

### 3.1 `examples/speculative-simple/speculative-simple.cpp`

这部分做了两类变化：

#### A. 删除旧的 example 层链表/槽位验证逻辑

删掉了之前 example 里自己维护的：

- `speculative_slot_chain`

它原来是：

- 通过 `llama_memory_seq_cp()` 把 live seq 复制到不同 seq slot
- 每个 depth 单独 `llama_decode()`
- 最后再 `promote_depth()`

这条路虽然能跑通，但本质上把 target 验证改成了串行，性能会明显变差。

#### B. 恢复成单次 batch verify

现在改成：

- 先调用 `llama_memory_eagle3_recurrent_begin(mem_tgt, 0, n_verify, n_past)`
- 然后把 `[id_last, draft0, draft1, ...]` 一次性塞进 `batch_tgt`
- 只做一次 `llama_decode(ctx_tgt, batch_tgt)`
- 再用 `common_sampler_sample_and_accept_n()` 一次性判断接受到哪个 depth
- 接着调用 `llama_memory_eagle3_recurrent_promote(mem_tgt, 0, ids.size())`
- 最后仍然用 `llama_memory_seq_rm(mem_tgt, 0, n_past, -1)` 清理多余 KV

#### C. 保留 `n_parallel = draft + 2`

这里保留了：

- `params.n_parallel = std::max(params.n_parallel, params.speculative.n_max + 2);`

原因不是再把额外 seq 当验证语义用，而是为了让 recurrent cache 物理上有足够 cell 可分配给临时 recurrent round-state。

#### D. 修复“首个生成词看起来没输出”的显示问题

之前 EAGLE3 路径里是：

- prompt decode 完后立刻 sample 出第一个生成 token
- 立刻 `LOG()` 打印这个 token
- 然后才执行 `common_speculative_init()`

问题在于：

- `common_speculative_init()` 会继续打印 draft context / reserve 等初始化日志
- 于是首个 token 会和后续日志粘在一行
- 看起来像“首词丢了”，实际是被日志冲掉显示了

现在改成：

- 先 sample 并 accept 第一个 token，但不立刻打印
- 用 `pending_first_token_print` 暂存
- 等 `common_speculative_init()` 和 `common_speculative_begin()` 完成后再打印

这样不改变采样和状态推进语义，只修输出顺序。

---

### 3.2 `include/llama.h`

新增了 3 个公开 API：

- `llama_memory_eagle3_recurrent_begin(...)`
- `llama_memory_eagle3_recurrent_promote(...)`
- `llama_memory_eagle3_recurrent_clear(...)`

作用：

- 给上层 example 提供“开始一轮 EAGLE3 recurrent 验证”
- “把接受的 depth 提升成 live recurrent state”
- “清理这一轮临时 recurrent 状态”

---

### 3.3 `src/llama-context.cpp`

这里有两类内容：

#### A. 新增上述 3 个 API 的 wrapper

具体是把 `llama_memory_t` 转发到 memory 对象：

- `mem->eagle3_recurrent_round_begin(...)`
- `mem->eagle3_recurrent_round_promote(...)`
- `mem->eagle3_recurrent_round_clear(...)`

#### B. 文件中还存在一段 EAGLE3 graph type 选择调整

当前 `graph_reserve()` 里有一段：

- 优先判断 `model.target_tok_embd != nullptr` 就走 decoder
- 否则走 encoder

这段目前也在当前 diff 里，但它不属于这次 recurrent round-state 主修复链路，而是当前工作区里同文件的现有改动。

---

### 3.4 `src/llama-memory.h`

在 `llama_memory_i` 抽象接口里新增了默认虚函数：

- `eagle3_recurrent_round_begin(...)`
- `eagle3_recurrent_round_promote(...)`
- `eagle3_recurrent_round_clear(...)`

默认实现是空操作 / 返回 `false`。

目的：

- 不破坏其他 memory 类型
- 只有 recurrent / hybrid 实际重载实现

---

### 3.5 `src/llama-memory-hybrid.h` 和 `src/llama-memory-hybrid.cpp`

这里新增了 hybrid 层的转发：

- `eagle3_recurrent_round_begin(...)`
- `eagle3_recurrent_round_promote(...)`
- `eagle3_recurrent_round_clear(...)`

全部直接转给：

- `mem_recr`

因为 EAGLE3 这次需要额外保存的是 recurrent state，不是 attn KV。

---

### 3.6 `src/llama-memory-recurrent.h`

这是这次改动最核心的头文件之一。

新增内容包括：

#### A. 对外接口

- `eagle3_recurrent_round_begin(...)`
- `eagle3_recurrent_round_promote(...)`
- `eagle3_recurrent_round_clear(...)`
- `eagle3_recurrent_round_active(...)`
- `eagle3_recurrent_round_cell(...)`

#### B. `mem_cell` 新增字段

在原来的：

- `pos`
- `src`
- `src0`
- `tail`
- `seq_id`

之外，新增了：

- `eagle3_owner`
- `eagle3_depth`

用于标记“这个 cell 当前是不是某个 live seq 的 EAGLE3 临时 depth cell”。

#### C. `is_empty()` 语义变更

原来只看：

- `seq_id.empty()`

现在改成同时要求：

- `seq_id.empty()`
- `eagle3_owner < 0`

这样临时 recurrent cell 不会被正常分配逻辑误认为是空的。

#### D. 新增一轮 round-state 数据结构

新增：

- `struct eagle3_recurrent_round_state`

里面存：

- `live_seq_id`
- `base_pos`
- `depth_cells`

其中：

- `depth_cells[depth - 1] = 对应 depth 的 recurrent cell id`

#### E. `llama_memory_recurrent_context` 新增查询接口

新增：

- `has_eagle3_round(...)`
- `get_eagle3_round_cell(...)`

这是给 Qwen3.5 图构建阶段使用的。

---

### 3.7 `src/llama-memory-recurrent.cpp`

这是这次改动最核心的实现文件。

#### A. `clear()` 增强

现在 `clear()` 会先：

- `eagle3_recurrent_round_clear_all()`

然后把每个 cell 的：

- `src0`
- `eagle3_owner`
- `eagle3_depth`

一起清空。

#### B. 新增临时 recurrent cell 分配/释放逻辑

新增私有函数：

- `eagle3_recurrent_cell_alloc()`
- `eagle3_recurrent_cell_reserve()`
- `eagle3_recurrent_cell_release()`
- `eagle3_recurrent_round_clear_all()`

作用：

- 在当前 recurrent memory 里，额外占用若干物理 cell
- 给一轮 EAGLE3 verify 的每个 depth 预留一个 cell
- 用完后释放

#### C. 新增 round 生命周期实现

新增：

- `eagle3_recurrent_round_begin(...)`
- `eagle3_recurrent_round_promote(...)`
- `eagle3_recurrent_round_clear(...)`

行为如下：

##### `begin()`

- 清理旧 round
- 为 `depth = 1..n_depth` 分配临时 recurrent cell
- 把这些 cell 记录到 `eagle3_rounds[live_seq_id].depth_cells`

##### `promote()`

- 找到接受的 `depth` 对应 cell
- 释放其他临时 depth cell
- 把被接受 cell 从“临时 EAGLE3 cell”转成“live seq 的真实 tail”
- 关键补丁：把接受 cell 的
  - `src = accepted_cell_id`
  - `src0 = accepted_cell_id`
  补上，避免下一轮把这个 state 误当成“从零开始”

##### `clear()`

- 释放该 live seq 当前 round 的所有临时 cell

#### D. 常规清理路径补齐 `src0`

在这些函数里，凡是把 cell 清空的地方，也一起清：

- `seq_rm()`
- `seq_cp()`
- `seq_keep()`

新增：

- `cells[i].src0 = -1`

避免残留旧的 source metadata。

#### E. context 查询实现

在末尾新增：

- `has_eagle3_round(...)`
- `get_eagle3_round_cell(...)`

供图构建时读取“本轮每个 depth 对应哪个 recurrent cell”。

---

### 3.8 `src/models/qwen35.cpp`

这是这次第二核心文件。

目标是：

- 当检测到当前 batch 是一次 EAGLE3 verify round
- 不再只写最终 recurrent state
- 而是把 batch 内每个 token 的中间 recurrent state 都落到各自的临时 cell

#### A. 识别当前 batch 是否有 EAGLE3 round

通过：

- `mctx_cur->has_eagle3_round(live_seq_id)`
- `mctx_cur->get_eagle3_round_cell(live_seq_id, t + 1)`

拿到：

- 每个 seq、每个 token depth 对应的 recurrent cell id

#### B. 保存 conv recurrent state

原来的实现只把：

- `last_conv_states`

整体写回当前 live 位置。

现在如果有 EAGLE3 round：

- 会对每个 `seq`
- 对每个 `token depth`
- 从 `conv_input` 里取出“消费该 token 后”的 conv window
- 写到对应临时 recurrent cell

这里后面又修了一个 bug：

- 偏移从 `t * sizeof(elem)` 改成了 `(t + 1) * sizeof(elem)`

否则取到的是“消费 token 前”的窗口，会把下一轮 recurrent state 带偏。

#### C. 保存 SSM recurrent state

原来的实现是：

- `build_delta_net(q, k, v, gate, beta, state, il)`
- 只得到一个最终 `new_state`
- 再一次性写回 `ssm_states_all`

现在如果有 EAGLE3 round：

- 对每个 token depth 单独切 `q/k/v/gate/beta`
- 逐 token 调 `build_delta_net(...)`
- 每一步得到新的 `state_cur`
- 把这个 `state_cur` 对应 seq 的切片写到对应临时 recurrent cell
- 同时把 `output` 用 `ggml_concat(..., dim=2)` 拼回去，保持整轮 batch 仍然是一个完整输出张量

这样：

- target 验证仍然是一次 batch
- 但 recurrent 中间状态也被真实保存下来了

#### D. 非 EAGLE3 round 时保持原逻辑

如果没有当前 round-state：

- 仍然走原来的单次 `build_delta_net()` + 单次 `new_state` 写回

---

### 3.9 `src/models/qwen35moe.cpp`

这里做了与 `qwen35.cpp` 同样的一套修改。

也就是说：

- conv state 的逐 depth 保存
- SSM recurrent state 的逐 depth 保存
- `output` 拼回 batch
- 非 EAGLE3 round 时退回原逻辑

这样 Qwen3.5 MoE 版本也同步具备同样行为。

---

## 4. 这次修复过程中额外修掉的两个关键 bug

### Bug 1：conv 中间状态偏移错误

问题：

- 一开始保存每个 depth 的 conv state 时，偏移用了 `t`
- 实际应该用 `t + 1`
- 否则保存的是“消费当前 token 之前”的窗口

修复：

- `qwen35.cpp`
- `qwen35moe.cpp`

里都改成了：

- `(t + 1) * ggml_element_size(conv_input)`

### Bug 2：promote 后 recurrent state 的 `src/src0` 没补

问题：

- 接受的临时 recurrent cell 转成 live cell 后
- 如果 `src/src0` 仍是 `-1`
- 下一轮 `find_slot()` 很可能把它当成“从零状态开始”
- 后果就是生成退化、重复输出

修复：

- 在 `llama_memory_recurrent::eagle3_recurrent_round_promote()` 中补上：
  - `accepted_cell.src  = accepted_cell_id`
  - `accepted_cell.src0 = accepted_cell_id`

这个修复对稳定输出很关键。

---

## 5. 当前运行状态

按当前代码，下面这条命令已经可以正常跑完：

```bash
/home/ljl/ljl_test/llama/build/bin/llama-speculative-simple \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf \
  -md /home/ljl/ljl_test/llama/models/eagle3-qwen35-4b-draft-Q4_K_M.gguf \
  --eagle3 -p "Write a function to remove all elements from a given list present in another list." \
  -n 256 --draft 2 --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99
```

已确认：

- 不再触发之前的 recurrent rollback 失败
- 不再触发之前的明显重复刷屏坏结果
- target 验证已恢复为一次 batch verify
- recurrent 中间状态已通过 core round-state 方案保存并 promote

---

## 6. 当前实现的设计结论

现在这版不是“在 example 里模拟链表 seq slot”。

现在的实现是：

- example 只负责发起 round begin / promote
- core memory 负责维护一轮 EAGLE3 的临时 recurrent cell
- Qwen3.5 graph 负责把 batch 内每个 depth 的 recurrent 中间状态真实写入这些 cell
- accept 后直接把接受 depth 对应 cell 提升为 live tail

这更接近最终想要的方向。

---

## 7. 当前还可以继续优化的点

虽然功能已经基本跑通，但还可以继续看：

- 为什么速度仍然可能不如预期
- 是否还有多余的 graph / state copy 开销
- 是否要把这套 round-state 进一步抽象成更通用的 recurrent depth slot 机制

---

## 8. 一句话总结

这轮修改已经把 EAGLE3 在 hybrid/recurrent 架构下最核心的问题解决到了 core 层：

- 保留 batch verify
- 保存每个 depth 的 recurrent 中间状态
- accept 后直接 promote 到 live recurrent state

不再依赖 example 层串行 replay 或链表 seq slot 模拟。
