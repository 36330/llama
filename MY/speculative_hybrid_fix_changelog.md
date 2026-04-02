# Speculative Decoding Hybrid 架构适配修改记录

## 当前代码现状

这份文档前面保留了完整排查历史，但如果只看“当前代码里还剩什么”，结论如下：

当前仍保留的代码改动:

1. `examples/speculative-simple/speculative-simple.cpp`

当前已经回退的代码改动:

1. `src/llama-memory-hybrid.cpp` 的非原子 `seq_rm`
2. `src/llama-batch.cpp` 的 nested subset split
3. `src/llama-kv-cache.cpp` 的 multi-seq mask patch
4. `common/sampling.cpp` 的调试日志

当前最终保留方案:

1. 不动核心层 memory / batch / kv / graph 逻辑
2. 只在 example 层做一个局部状态链 patch
3. 用 `speculative_slot_chain` 记录 `live -> id_last -> draft0 -> draft1 -> ...`
4. reject 时直接 promote 最后 accepted 的链节点

当前结论:

1. 正确性比之前好，坏输出不再是主要问题
2. 侵入性已经降到很低
3. 性能明显不行，这是当前最主要问题

当前变慢的直接原因:

1. verify 阶段现在是逐 token decode
2. 每一个 `id_last / draft0 / draft1 / ...` 都单独走一次 `llama_decode`
3. speculative 本来依赖“批量验证多个 token”吃吞吐
4. 当前版本把这个优势基本打碎了

## 问题

Qwen3.5 是 hybrid 架构（KV cache + Gated Delta Net recurrent state），speculative decoding 验证后调用 `seq_rm(0, n_past, -1)` 清理 rejected draft tokens 时：

1. `llama_memory_recurrent::seq_rm` 不支持部分删除 → 返回 false
2. 原 `llama_memory_hybrid::seq_rm` 是原子操作 → recurrent 失败后 KV cache 也不清理
3. 下一轮 `llama_decode` 因 M-RoPE 位置冲突 → 返回 -1

## 方案：快照 + 非原子 seq_rm + O(k) 部分重放

---

## 修改 1: `src/llama-memory-hybrid.cpp` 第 132-139 行

**改动**: `seq_rm` 从原子操作改为非原子，KV cache 和 recurrent 独立执行

**修改前**:
```cpp
bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}
```

**修改后**:
```cpp
bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Always clean up both caches independently.
    // Recurrent partial deletion may fail (e.g. speculative decoding rollback),
    // but KV cache should still be cleaned to avoid position conflicts.
    bool recr_ok = mem_recr->seq_rm(seq_id, p0, p1);
    bool attn_ok = mem_attn->seq_rm(seq_id, p0, p1);
    return recr_ok && attn_ok;
}
```

**效果**:
- `seq_rm(0, n_past, -1)` → KV cache 正确删除 rejected tokens，recurrent 失败但无副作用
- 返回 false 通知调用方 recurrent state 需要修复
- 对纯 KV 模型/全删操作无影响

---

## 修改 2: `examples/speculative-simple/speculative-simple.cpp`

### 2a. 循环外预分配快照 buffer（第 191-194 行，原 189 行之后插入）

```cpp
// [HYBRID] pre-allocate recurrent state snapshot buffer (size=0 for pure KV models, no overhead)
llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
const size_t recr_state_size = llama_state_seq_get_size_ext(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
std::vector<uint8_t> recr_snapshot(recr_state_size);
```

**说明**: 纯 KV cache 模型 `recr_state_size = 0`，vector 不分配内存，零开销。

### 2b. 循环内 draft 前保存快照（第 210-215 行，原 218 行之前插入）

```cpp
// [HYBRID] save recurrent state snapshot before draft+verify cycle
if (recr_state_size > 0) {
    llama_state_seq_get_data_ext(ctx_tgt, recr_snapshot.data(), recr_snapshot.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
}
const llama_token id_last_saved = id_last;
const int n_past_saved = n_past;
```

**说明**: 在 draft 生成和 target decode 之前保存 recurrent state 快照及当前状态，用于后续可能的回滚。

### 2c. 替换 seq_rm 为快照+重放 fallback（第 289-313 行，替换原 290 行）

**修改前**:
```cpp
llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
```

**修改后**:
```cpp
if (!llama_memory_seq_rm(mem_tgt, 0, n_past, -1)) {
    // [HYBRID] KV cache already cleaned (non-atomic seq_rm), but recurrent state has draft pollution
    if (recr_state_size > 0) {
        // 1) restore recurrent state to pre-decode snapshot
        llama_state_seq_set_data_ext(ctx_tgt, recr_snapshot.data(),
            recr_snapshot.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        // 2) delete KV cache entries for accepted tokens (make room for replay)
        llama_memory_seq_rm(mem_tgt, 0, n_past_saved, n_past);

        // 3) batch replay accepted tokens to rebuild both KV cache + recurrent state
        const int n_replay = (int)ids.size();
        llama_batch batch_replay = llama_batch_init(n_replay, 0, 1);
        common_batch_clear(batch_replay);
        common_batch_add(batch_replay, id_last_saved, n_past_saved, {0}, n_replay == 1);
        for (int i = 0; i < n_replay - 1; i++) {
            common_batch_add(batch_replay, draft[i], n_past_saved + 1 + i, {0}, i == n_replay - 2);
        }
        llama_decode(ctx_tgt, batch_replay);
        llama_batch_free(batch_replay);

        LOG_DBG("[HYBRID] replayed %d accepted tokens to rebuild recurrent state\n", n_replay);
    }
}
```

**说明**: 同时删除了之前注释掉的 Option C 全量重建代码（原 292-313 行）。

---

## 位置追踪验证

```
decode 前:  n_past_saved = P,  KV max = P-1,  recr pos = P-1
decode:     batch = [id_last@P, d0@P+1, ..., d7@P+8]
            KV max = P+8, recr pos = P+8
验证:       accept k 个 → n_past = P + k

seq_rm(0, P+k, -1):  返回 false
  attn:     删除 [P+k, ∞) → KV max = P+k-1  ✓
  recr:     部分删除失败 → recr pos 仍 = P+8

fallback:
  恢复快照 → recr pos = P-1
  seq_rm(0, P, P+k) → 删除 KV [P, P+k), 成功
  重放 k 个 token @pos [P, P+k-1] → KV + recurrent 都正确写入

结束: KV max = P+k-1, recr pos = P+k-1, n_past = P+k  ✓ 一致
```

## 性能

| 场景 | 额外开销 |
|------|---------|
| 纯 KV cache 模型 | 零（recr_state_size=0，无快照，seq_rm 成功） |
| hybrid + 全部 accept | 快照保存（~几百KB memcpy），seq_rm 成功无回滚 |
| hybrid + 部分 reject | 快照保存 + 恢复 + 1 次 k-token batch decode |

k 通常 1-5，远小于 seq_len，比之前 Option C 的 O(seq_len) 全量重建快几个数量级。

## 测试命令

```bash
cd /home/ljl/ljl_test/llama/build && \
./bin/llama-speculative-simple \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf \
  -md /home/ljl/ljl_test/llama/models/eagle3-qwen35-4b-draft-Q4_K_M.gguf \
  --eagle3 -p "Write a function to remove all elements from a given list present in another list." \
  -n 256 --draft 8 --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99
```

## 追加修正: 放弃 nested multi-seq verify，改成逐 slot 链式 verify

前一版把 target verify 做成了一个 nested multi-seq batch，目的是一次 decode 出所有 draft 对应的中间状态。

实际运行后发现这条路对 hybrid / recurrent 模型不成立，原因不是只在 KV mask：

1. `llama_memory_recurrent` 仍然是“每个序列一个旧状态入口”的语义。
2. 同一批里的多个 verify token 会并行读取旧状态，不会把前一个 draft 的新 recurrent state 串给后一个 draft。
3. 结果就是状态链断掉，输出会出现明显异常。

因此这次把 `examples/speculative-simple/speculative-simple.cpp` 的 verify 主循环改成：

1. `seq 0` 保持 live state
2. `seq 1` = 基于 `seq 0` decode `id_last`
3. `seq 2` = 基于 `seq 1` decode `draft0`
4. `seq 3` = 基于 `seq 2` decode `draft1`
5. 以此类推

这样每个 slot 都对应一个真实中间 recurrent state。

验证完成后：

1. 如果 accept 到第 `k` 个位置，就直接 promote `seq k+1 -> seq 0`
2. 不需要 snapshot + replay
3. 也不依赖 nested multi-seq recurrent batch 的隐式链式传播

## 追加验证结果

用上面的测试命令实跑后：

1. 之前那种 `20user 20user ...` 的明显坏输出没有再复现。
2. 当前仍然会看到部分初始化日志夹在生成文本之间，这主要是日志输出时序问题，不再是 recurrent state 串错导致的 token 污染。

## 最新收口版本

后面又按“最小程度修改、不影响原功能”的要求继续收口了一轮。

当前保留生效的 speculative / hybrid 修复方式已经不是前面文档里的那些核心层 patch，而是:

1. 只在 `examples/speculative-simple/speculative-simple.cpp` 里加一个本地 `speculative_slot_chain`
2. 用这个链表式状态结构管理 `live -> id_last -> draft0 -> draft1 -> ...`
3. reject 时直接按链深度找到最后接受的状态并 promote 回 `seq 0`

已经回退的内容:

1. `src/llama-memory-hybrid.cpp` 的非原子 `seq_rm`
2. `src/llama-batch.cpp` 的 nested subset sequential split
3. `src/llama-kv-cache.cpp` 的 multi-seq mask patch

也就是说，当前版本的设计取向是:

1. 不改底层 memory / kv / batch 默认语义
2. 不 monkey patch 核心类
3. 只在 example 层面做一层局部状态链管理

## 当前性能结论

这版“逻辑正确”，但速度退化明显。

原因:

1. 现在 verify 阶段是逐 token decode
2. 每个 `id_last / draft0 / draft1 / ...` 都要单独走一次 `llama_decode`
3. recurrent state 保持正确了，但把原来 speculative 一次验证多 token 的吞吐优势打碎了

所以当前状态可以概括成:

1. 正确性: 已经比前面的错误版本好，坏输出不再复现
2. 侵入性: 已经降到最小，只改一个 example 文件
3. 性能: 还不行，下一步需要专门讨论如何把这条正确路径重新批量化
