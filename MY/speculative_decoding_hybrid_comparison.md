# Speculative Decoding: Hybrid架构中Recurrent State回退问题

## 当前代码对应关系

这份文档描述的是“理想槽位方案”和 llama.cpp 原始 recurrent 语义之间的差异。

当前代码并没有实现这里说的“真正的底层槽位方案”，而是处在一个更保守的中间状态：

1. 没有修改 `llama_memory_recurrent` 的 core cell 布局
2. 没有给 recurrent memory 增加 `num_spec + 1` 个底层永久槽位
3. 没有修改 recurrent graph / kernel 让它在一次 batch 里天然写出每个 draft 的中间状态

当前实际代码做的是:

1. 只在 `examples/speculative-simple/speculative-simple.cpp` 中构造一个“逻辑上的状态链”
2. 用多个 seq slot 模拟:
   `live -> id_last -> draft0 -> draft1 -> ...`
3. 每个 token 单独 decode 一次，拿到下一个真实状态
4. reject 时直接回到链上最后 accepted 的那个状态

所以当前代码状态可以这样理解:

1. 不是文档里理想的 O(1) 底层槽位写入方案
2. 也不是最早的 snapshot + replay 回退方案
3. 而是一个“example 层最小 patch 的链式状态方案”

它的优点是:

1. 最小改动
2. 不破坏原功能
3. 逻辑清楚

它的缺点是:

1. 速度慢
2. 因为每个 verify token 都单独 decode
3. 所以它更适合作为“正确性基线”，不适合作为最终性能版本

## 问题描述

在EAGLE3/Speculative Decoding中，当验证draft tokens时，部分tokens会被rejected。对于：

- **纯KV cache模型**（如Llama）：可以直接删除rejected tokens的KV cache
- **Hybrid模型**（如Qwen3.5，包含KV cache + Recurrent state）：Recurrent state（Mamba/delta-net）是**累积状态**，无法部分删除

## 核心问题

```
Recurrent state的累积特性：
State[0] = f(prompt)
State[1] = f(prompt, token0) = f(State[0], token0)
State[2] = f(prompt, token0, token1) = f(State[1], token1)
...
```

每个状态依赖之前所有状态，不能像KV cache那样随意删除中间的token。

---

## vLLM的解决方案

### 核心策略：延迟清理 + 槽位机制

vLLM不采用"删除"策略，而是采用"覆盖"策略：

### 1. 状态槽位分配

```
每个sequence分配固定数量的状态槽位：num_spec + 1

block_table[:, :num_spec + 1]
┌────────┬────────┬────────┬────────┬────────┐
│ slot_0 │ slot_1 │ slot_2 │ slot_3 │ slot_4 │
│ (base) │ draft0 │ draft1 │ draft2 │ draft3 │
└────────┴────────┴────────┴────────┴────────┘
```

### 2. 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ 第一次Draft: 生成5个tokens                                   │
├─────────────────────────────────────────────────────────────┤
│ b_h初始值 = State[-1] = 包含整个prompt的状态                  │
├─────────────────────────────────────────────────────────────┤
│ i_t=0: b_h = f(初始状态, token0) → 保存到state[0]            │
│ i_t=1: b_h = f(state[0], token1) → 保存到state[1]           │
│ i_t=2: b_h = f(state[1], token2) → 保存到state[2]           │
│ ...                                                         │
└─────────────────────────────────────────────────────────────┘

验证结果: token0✓ token1✓ token2✗ token3✗ token4✗
           num_accepted = 2

┌─────────────────────────────────────────────────────────────┐
│ 第二次Draft:                                                │
├─────────────────────────────────────────────────────────────┤
│ b_h初始值 = state[1] = prompt + token0 + token1 (完整状态!)  │
├─────────────────────────────────────────────────────────────┤
│ i_t=0: b_h = f(state[1], new_token0) → 覆盖state[0]        │
│ i_t=1: b_h = f(state[0], new_token1) → 覆盖state[1]        │
│ ...                                                         │
└─────────────────────────────────────────────────────────────┘
```

### 3. 关键Kernel实现

```python
# vllm/model_executor/layers/fla/ops/fused_recurrent.py

# 读取初始状态：从accepted的最后一个位置开始
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    p_h0 = h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)

# 增量计算每个draft token
for i_t in range(0, T):
    b_h = compute_new_state(b_h, current_token)
    # 保存到对应槽位
    store(ssm_state[seq_idx, i_t], b_h)
```

### 4. PADDING_SLOT_ID机制

```python
PADDING_SLOT_ID = -1

# 被拒绝的位置不会被写入，避免污染
slot_mapping[rejected_mask] = PADDING_SLOT_ID
```

---

## llama.cpp的处理方式及问题

### 当前实现

```cpp
// llama-memory-recurrent.cpp

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // ...
    // models like Mamba or RWKV can't have a state partially erased
    if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
        return false;  // 无法部分删除！
    }
    // ...
}
```

### 问题分析

| 方面 | llama.cpp | 问题 |
|------|-----------|------|
| 状态存储 | 每个seq只有1个cell | 无法保留中间状态 |
| 删除策略 | 部分删除（失败） | 返回false |
| 回退机制 | 无 | 需要上层fallback |
| 性能 | O(seq_len)重建 | 浪费计算 |

### 你尝试的Fallback方案

```cpp
// 保存recurrent state快照
llama_state_seq_get_data_ext(ctx_tgt, recr_state_buf.data(), ...);

// 尝试删除
if (!llama_memory_seq_rm(mem_tgt, 0, n_past, -1)) {
    // 删除失败，回退并重建
    llama_state_seq_set_data_ext(ctx_tgt, recr_state_buf.data(), ...);
    llama_memory_seq_rm(mem_tgt, 0, -1, -1);  // 清空全部

    // 重新decode整个prompt_tgt重建状态
    for (size_t i = 0; i < prompt_tgt.size(); i += n_batch_size) {
        llama_decode(ctx_tgt, llama_batch_get_one(prompt_tgt.data() + i, n));
    }
}
```

**问题**：每次fallback都要O(seq_len)重新计算，性能损失大。

---

## 对比总结

| 方面 | llama.cpp (当前) | vLLM |
|------|------------------|------|
| **策略** | 删除失败 → 完全重建 | 延迟清理 → 覆盖 |
| **状态粒度** | 每个seq 1个cell | 每个seq (num_spec+1) 个槽位 |
| **删除操作** | seq_rm() 尝试删除 | 不删除，标记num_accepted |
| **回退开销** | O(seq_len) 重建 | O(1) 索引跳转 |
| **状态连续性** | 重建后恢复 | 增量累积，始终连续 |
| **实现复杂度** | fallback逻辑复杂 | kernel内部处理 |

---

## 解决方案：为llama.cpp添加槽位机制

### 核心思路

1. **不删除**：Rejected tokens的状态保留，下次迭代自然覆盖
2. **槽位管理**：每个sequence分配 `num_spec + 1` 个连续cell
3. **索引跳转**：通过 `num_accepted` 直接跳转到正确的状态

### 代码修改方案

#### 1. 扩展 mem_cell 结构体

```cpp
struct mem_cell {
    llama_pos pos  = -1;
    int32_t   src  = -1;
    int32_t   src0 = -1;
    int32_t   tail = -1;
    std::set<llama_seq_id> seq_id;

    // 新增：speculative decoding 支持
    bool is_spec_slot = false;     // 是否是speculative槽位
    int32_t spec_slot_idx = -1;    // 槽位索引 (0=base, 1..n=draft slots)
    int32_t spec_base_seq = -1;    // 所属的基础seq_id
    bool spec_is_accepted = false; // 此槽位是否被accept
};
```

#### 2. 添加speculative管理成员

```cpp
class llama_memory_recurrent {
    // 新增：speculative decoding 配置
    bool spec_enabled = false;
    uint32_t spec_n_slots = 0;              // 每个seq的draft槽数量
    std::vector<int32_t> spec_num_accepted; // 每个seq的accepted数量
};
```

#### 3. 添加新的API

```cpp
// 启用speculative模式
void spec_enable(uint32_t n_slots);

// 初始化一个seq的speculative槽位
void spec_init_seq(llama_seq_id seq_id);

// 更新accepted数量（验证后调用）
void spec_update_accepted(llama_seq_id seq_id, int32_t n_accepted);

// 获取seq的初始状态cell索引
int32_t spec_get_initial_cell(llama_seq_id seq_id) const;

// 获取seq的第slot_idx个槽位的cell索引
int32_t spec_get_slot_cell(llama_seq_id seq_id, int32_t slot_idx) const;

// spec模式下尝试跳过seq_rm
bool spec_try_skip_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
```

#### 4. 修改 seq_rm

```cpp
bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // spec模式下，尝试跳过删除
    if (spec_try_skip_rm(seq_id, p0, p1)) {
        return true;  // 已处理，跳过原有删除逻辑
    }

    // 原有逻辑保持不变
    // ...
}
```

#### 5. 上层调用修改

```cpp
// 初始化
llama_memory_recurrent * mem = ...;
mem->spec_enable(params_speculative.n_max);  // n_max是draft数量
mem->spec_init_seq(0);  // seq_id = 0

// 主循环
while (true) {
    // 生成draft
    llama_tokens draft = common_speculative_draft(...);

    // Decode
    common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
    for (size_t i = 0; i < draft.size(); ++i) {
        common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
    }
    llama_decode(ctx_tgt, batch_tgt);

    // 验证
    const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
    size_t n_accepted = ids.size() - 1;

    // 更新speculative状态（关键！）
    mem->spec_update_accepted(0, n_accepted);

    // 更新prompt
    for (size_t i = 0; i < ids.size(); ++i) {
        prompt_tgt.push_back(id_last);
        id_last = ids[i];
    }
    n_past += n_accepted;

    // 不再需要 llama_memory_seq_rm！
    // 下次decode会自动从正确的槽位继续
}
```

---

## 内存布局

```
原: cells = [cell_0, cell_1, cell_2, ...]  // 每个seq一个cell

新: cells = [seq_0_base, seq_0_slot_1, seq_0_slot_2, ...,  // seq_0的槽位组
             seq_1_base, seq_1_slot_1, seq_1_slot_2, ...,  // seq_1的槽位组
             ...]

每个seq分配 (num_max_drafts + 1) 个连续cell
```

---

## 实现优势

1. **向后兼容**：原有代码完全不受影响
2. **职责分离**：speculative逻辑独立管理
3. **性能提升**：O(1)索引跳转 vs O(seq_len)重建
4. **易于调试**：每个槽位的状态一目了然
5. **无需重建**：不需要保存/恢复状态快照

---

## 关键结论

1. **vLLM不删除，只覆盖**：通过槽位机制保留中间状态，自然覆盖
2. **状态始终累积**：每个槽位保存从prompt开始的完整累积状态
3. **O(1)回退**：通过num_accepted索引直接跳转，不需要重建
4. **llama.cpp需要类似机制**：添加槽位管理，避免fallback重建
