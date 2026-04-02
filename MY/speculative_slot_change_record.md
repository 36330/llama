# Speculative Slot Change Record

这份记录覆盖我这次在工作区里手工改过、且和这轮 speculative / hybrid 排查直接相关的内容。

## Current Code Status

下面这部分只描述“当前还保留在代码里的状态”，不描述中途试过又回退的方案。

当前实际仍保留的源码改动:

1. `examples/speculative-simple/speculative-simple.cpp`

当前已经回退、不再保留在代码里的改动:

1. `common/sampling.cpp`
2. `src/llama-batch.cpp`
3. `src/llama-kv-cache.cpp`
4. `src/llama-memory-hybrid.cpp`
5. 其它 speculative 相关核心层 monkey patch

当前实现目标:

1. 不改 `llama_memory_recurrent` 的核心数据结构
2. 不改 `find_slot()` / graph / recurrent kernel 语义
3. 只在 `speculative-simple` 里用一个局部 helper 管理 hybrid draft state

当前保留实现的核心做法:

1. 仍然强制为 target 预留额外 seq slot:
   `params.kv_unified = true;`
   `params.n_parallel = max(params.n_parallel, params.speculative.n_max + 2);`
2. 新增本地结构 `speculative_slot_chain`
3. 每个 verify token 单独 decode 一次
4. 每次 decode 前，把上一个已确认状态复制到下一个 slot
5. 如果第 `k+1` 个位置 reject，则链上深度 `k` 的节点天然就是最后 accepted state
6. 直接 `promote_depth(ids.size())` 回到 live `seq 0`

当前本地 helper 结构:

```cpp
struct speculative_slot_chain {
    struct node {
        llama_seq_id seq_id = -1;
        int prev = -1;
        int depth = 0;
    };

    llama_memory_t mem = nullptr;
    llama_seq_id live_seq = 0;
    int max_slots = 0;
    int tail = -1;
    std::vector<node> nodes;

    void reset();
    llama_seq_id append();
    const node & find_depth(int depth) const;
    void promote_depth(int depth);
};
```

当前 verify 流程:

1. `slot_chain.reset()`
2. `append()` 出 `seq 1`，decode `id_last`
3. `append()` 出 `seq 2`，decode `draft0`
4. `append()` 出 `seq 3`，decode `draft1`
5. ...
6. 采样一旦和 draft 不一致就停止
7. `slot_chain.promote_depth(ids.size())`

当前优点:

1. 正确性已经明显好于前面错误版本
2. 只改一个 example 文件
3. 不会污染 llama.cpp 其它默认路径

当前缺点:

1. 很慢
2. verify 从“一次 target decode 验证多个 token”退化成“每个 token 一次 decode”
3. graph 调度 / kernel launch / synchronize 开销被放大
4. 所以当前版本可能比不开 EAGLE 还慢

历史上在这轮排查中手工改过的源码文件:

1. `common/sampling.cpp`
2. `examples/speculative-simple/speculative-simple.cpp`
3. `examples/speculative/speculative.cpp`
4. `src/llama-batch.cpp`
5. `src/llama-context.cpp`
6. `src/llama-kv-cache.cpp`
7. `src/llama-memory-hybrid.cpp`

当前仍然保留生效的源码改动:

1. `examples/speculative-simple/speculative-simple.cpp`

新增或补写的文档文件:

1. `MY/speculative_slot_change_record.md`（本文件）
2. `MY/speculative_hybrid_fix_changelog.md`
3. `MY/speculative_decoding_hybrid_comparison.md`

说明:

1. `build/` 下的大量变化是编译产物，不是手工逐处编辑，所以这里不展开逐项记录。
2. 这份文档前半部分保留“排查过程中曾经出现过的修改”，文末 `## 11. 最新收口状态` 记录当前最终保留的最小 patch。

构建验证:

```bash
cmake --build /home/ljl/ljl_test/llama/build --target llama-speculative-simple -j4
```

结果: 通过。

## 1. `examples/speculative-simple/speculative-simple.cpp`

### 1.1 新增头文件

原因: slot verify 需要 `std::max`。

改前:

```cpp
#include "llama.h"
#include "chat.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
```

改后:

```cpp
#include "llama.h"
#include "chat.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
```

### 1.2 强制为 target 预留 speculative slot

原因: 把 `seq 0` 作为 live sequence，`seq 1..draft+1` 作为本轮 verify 的临时槽位。

改前:

```cpp
    common_init();

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
```

改后:

```cpp
    common_init();

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // Reserve extra target slots for speculative slot chains:
    //   seq 0 = live sequence
    //   seq 1..draft+1 = [id_last, draft0, ...] verification slots
    params.kv_unified = true;
    params.n_parallel = std::max(params.n_parallel, params.speculative.n_max + 2);

    // init llama.cpp
```

### 1.3 新增 acceptance 分布统计数组

原因: 临时加了每轮接受长度统计输出。

改前:

```cpp
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;
```

改后:

```cpp
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_seq_accept[10] = {0};
    // used to determine end of generation
    bool has_eos = false;
```

### 1.4 target batch 从单 seq 改成多 seq slot batch

原因: 原来这里只给 target 一个 seq。
slot 方案需要 target batch 能携带多个临时 seq。

改前:

```cpp
    common_speculative_begin(spec, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();
```

改后:

```cpp
    common_speculative_begin(spec, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, llama_n_seq_max(ctx_tgt));

    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    const int n_slot_verify_max = params.speculative.n_max + 1;

    if ((int) llama_n_seq_max(ctx_tgt) < n_slot_verify_max + 1) {
        LOG_ERR("%s: target context has only %u sequence slots, need at least %d\n",
                __func__, llama_n_seq_max(ctx_tgt), n_slot_verify_max + 1);
        return 1;
    }

    const auto t_enc_end = ggml_time_us();
```

### 1.5 删除原来的单序列 verify decode

原因: 原方案是:

1. 把 `id_last` 加到 `seq 0`
2. 再把 `draft[i]` 也都加到 `seq 0`
3. 失败后通过 `snapshot + replay` 修正 recurrent state

现在改成 slot verify，不再 replay，也不再走“嵌套多 seq 一次 batch verify”，而是逐 token 沿 slot 链推进。

改前:

```cpp
        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            llama_decode(ctx_tgt, batch_tgt);
        }
```

改后:

```cpp
        // do not waste time on small drafts
        if (draft.size() < (size_t) params_spec.n_min) {
            draft.clear();
        }

        std::vector<llama_token> verify_tokens;
        verify_tokens.reserve(draft.size() + 1);
        verify_tokens.push_back(id_last);
        verify_tokens.insert(verify_tokens.end(), draft.begin(), draft.end());

        const int n_verify = verify_tokens.size();

        GGML_ASSERT(n_verify <= n_slot_verify_max);

        // Build the speculative chain one token at a time:
        //   seq 1 = state after id_last
        //   seq 2 = state after id_last, draft0
        //   seq 3 = state after id_last, draft0, draft1
        //   ...
        // On accept we can promote the matching slot directly to seq 0.
        llama_memory_seq_keep(mem_tgt, 0);

        llama_tokens ids;
        ids.reserve(n_verify);

        llama_seq_id seq_prev = 0;

        for (int i = 0; i < n_verify; ++i) {
            const llama_seq_id seq_cur = i + 1;

            llama_memory_seq_cp(mem_tgt, seq_prev, seq_cur, -1, -1);

            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, verify_tokens[i], n_past + i, { seq_cur }, true);

            llama_decode(ctx_tgt, batch_tgt);

            const llama_token id = common_sampler_sample(smpl, ctx_tgt, 0);
            common_sampler_accept(smpl, id, true);

            ids.push_back(id);
            seq_prev = seq_cur;

            if ((size_t) i == draft.size()) {
                break;
            }

            if (draft[i] != id) {
                break;
            }
        }
```

### 1.6 采样逻辑从批量 row 取样改成逐 slot 立即取样

原因: 现在每一步只 decode 一个 token 到一个 slot，采样直接固定取当前 decode 的第 0 行 logits。

改前:

```cpp
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
```

改后:

```cpp
            const llama_token id = common_sampler_sample(smpl, ctx_tgt, 0);
            common_sampler_accept(smpl, id, true);

            ids.push_back(id);

            if ((size_t) i == draft.size()) {
                break;
            }

            if (draft[i] != id) {
                break;
            }
```

### 1.7 `n_past` 和 speculative accept 统计调整

原因: slot verify 里 `id_last` 也在本轮一起落进 slot，所以 `n_past` 增量从 `ids.size() - 1` 改成 `ids.size()`。
同时把 `common_speculative_accept()` 接上。

改前:

```cpp
        n_past    += ids.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();
```

改后:

```cpp
        n_past    += ids.size();
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_seq_accept[(ids.size()-1)] += 1;
        n_predict += ids.size();

        common_speculative_accept(spec, ids.size() - 1);
```

### 1.8 删除 snapshot+replay rollback，改成 slot promote

原因: 原来 hybrid/recurrent 失败时靠回放 accepted token 修正状态。
slot 方案下改成直接保留被接受位置的 slot，然后 promote 到 `seq 0`。

改前:

```cpp
        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }
```

改后:

```cpp
        const llama_seq_id s_keep = (llama_seq_id) ids.size();
        LOG_DBG("promote speculative slot %d to live sequence 0\n", (int) s_keep);

        llama_memory_seq_cp(mem_tgt, s_keep, 0, -1, -1);
        llama_memory_seq_keep(mem_tgt, 0);
```

### 1.9 新增 `n_seq_accept` 输出

原因: 输出每个接受长度的次数。

改前:

```cpp
    LOG_INF("n_draft   = %d\n", params_spec.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
```

### 1.10 提前初始化 speculator，避免 draft 上下文日志插进首个输出 token

原因: `--eagle3` 路径下如果先打印首个 target token，再初始化 speculative draft context，draft 的 `llama_context` / `sched_reserve` 日志会插到正文里。

改前:

```cpp
    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // Tokenize the prompt
```

改后:

```cpp
    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
    const auto & params_spec = params.speculative;
    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    // Tokenize the prompt
```

改前:

```cpp
    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    common_speculative_begin(spec, prompt_tgt);
```

改后:

```cpp
    common_speculative_begin(spec, prompt_tgt);
```

### 1.11 流式输出后立刻 `fflush(stdout)`

原因: 让 stdout 上的 token 输出尽快落到终端，减少和 stderr 日志的拼接感。

改前:

```cpp
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }
```

改后:

```cpp
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }
    fflush(stdout);
```

改前:

```cpp
        LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
        n_predict++;
```

改后:

```cpp
        LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
        fflush(stdout);
        n_predict++;
```

改前:

```cpp
            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }
```

改后:

```cpp
            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
            fflush(stdout);
        }
```

改后:

```cpp
    LOG_INF("n_draft   = %d\n", params_spec.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    
    for (int j = 0; j < 10; j++)
    {
        
        LOG_INF("n_seq_accept[%d] accpted_num = %d\n", j,n_seq_accept[j]);
    }
    
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
```

## 2. `src/llama-batch.cpp`

### 2.1 放宽 `split_equal()` 对 sequential split 的限制

原因: slot verify batch 里的 seq-set 不是互不相交的，而是嵌套链:

```text
{1,2,3,4}
{2,3,4}
{3,4}
{4}
```

原逻辑只允许非重叠集合，所以这种 batch 会被拆散。

改前:

```cpp
llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch, bool sequential) {
    if (sequential && has_cpl) {
        LLAMA_LOG_ERROR("%s: sequential split is not supported when there are coupled sequences in the input batch (you may need to use the -kvu flag)\n", __func__);

        return {};
    }

    std::vector<seq_set_t> cur_seq_set;

    llama_seq_id last_seq_id = -1;

    // determine the non-overlapping sequence sets participating in this ubatch
```

改后:

```cpp
llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch, bool sequential) {
    std::vector<seq_set_t> cur_seq_set;

    llama_seq_id last_seq_id = -1;

    auto is_seq_set_subset = [](const seq_set_t & lhs, const seq_set_t & rhs) {
        return (lhs & ~rhs).none();
    };

    // determine the sequence sets participating in this ubatch
    // in sequential mode, we also allow a nested chain of overlapping sets:
    //   {1,2,3}, {2,3}, {3}, ...
    // this is used by speculative decoding slot batches with unified KV.
```

改前:

```cpp
        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            // no overlap with existing sequence sets:
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }
```

改后:

```cpp
        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            const bool overlaps = !(cur_seq_set[s] & seq_set[i]).none();
            if (!overlaps) {
                continue;
            }

            // Allow nested subset chains in sequential mode, otherwise require disjoint sets.
            if (!sequential || !is_seq_set_subset(seq_set[i], cur_seq_set[s])) {
                add = false;
                break;
            }
        }
```

## 3. `src/llama-kv-cache.cpp`

### 3.1 修复 KQ mask 只看第一个 `seq_id` 的问题

这是后续追加修复。

原因: 我把 `speculative-simple` 改成 nested multi-`seq_id` verify batch 以后，
这里如果仍然只看 `ubatch->seq_id[i][0]`，就会把其它 slot 的可见历史丢掉，
实际运行会出现明显错误输出。

### 3.2 `seq_pos_min` 从单 `seq_id` 改成遍历全部 `seq_id`

改前:

```cpp
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }
```

改后:

```cpp
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            const llama_seq_id seq_id = ubatch->seq_id[i][k];
            seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
        }
    }
```

### 3.3 引入 `n_seq_id / seq_id0 / multi_seq`

原因: 对多 `seq_id` token 单独处理。

改前:

```cpp
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);
```

改后:

```cpp
            const int32_t n_seq_id = ubatch->n_seq_id[i];
            const llama_seq_id seq_id0 = ubatch->seq_id[i][0];
            const bool multi_seq = n_seq_id > 1;

            const auto & cells = v_cells.at(seq_to_stream[seq_id0]);
```

### 3.4 mask 复用优化只在单 `seq_id` token 上启用

原因: 原来的 mask copy/reuse 优化默认“一条 token 只对应一个 sequence”。
多 `seq_id` token 上继续复用会带入错误 mask。

改前:

```cpp
            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];
```

改后:

```cpp
            auto & idxs = seq_idxs[seq_id0];

            if (!alibi && !multi_seq) {
                if (seq_srct.find(seq_id0) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id0];
```

改前:

```cpp
                    seq_srct[seq_id] = i;
```

改后:

```cpp
                    seq_srct[seq_id0] = i;
```

### 3.5 可见性检查从单 `seq_id` 改成“任一 `seq_id` 命中即可”

这是这次运行错误的核心修复。

改前:

```cpp
                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }
```

改后:

```cpp
                bool visible = false;

                // mask the token if the KV cell is not visible from any of this token's sequence ids
                for (int32_t k = 0; k < n_seq_id; ++k) {
                    if (cells.seq_has(j, ubatch->seq_id[i][k])) {
                        visible = true;
                        break;
                    }
                }
                if (!visible) {
                    goto skip;
                }
```

### 3.6 `idxs.push_back()` 的窗口记录只保留单 `seq_id` 路径

改前:

```cpp
                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }
```

改后:

```cpp
                if (!alibi && !multi_seq) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id0]) {
                            idxs.push_back(j);
                        }
                    }
                }
```

## 4. 本次没有改动但被间接废弃的旧路径

`speculative-simple` 里之前那套:

1. 保存 recurrent snapshot
2. `seq_rm()` 失败后恢复 snapshot
3. replay accepted tokens

这次没有再保留；新的 `speculative-simple` 已经完全切成 slot promote 路径。

## 5. 总结

这次改动分成两轮:

1. 先把 `speculative-simple` 改成 slot verify + slot promote，并补 `llama-batch.cpp` 让嵌套 seq-set 能进同一个 ubatch。
2. 运行后发现输出异常，再补 `llama-kv-cache.cpp`，把 KQ mask 从“只看第一个 `seq_id`”修成“支持多 `seq_id` token”。

如果后面还要继续查 slot verify 行为问题，优先看:

1. `examples/speculative-simple/speculative-simple.cpp`
2. `src/llama-batch.cpp`
3. `src/llama-kv-cache.cpp`

## 6. `common/sampling.cpp`

### 4.1 新增一条 speculative-simple 调试日志

原因: 临时确认“草稿 token 全部命中后再采最后一个 target token”的分支是否被走到。

改前:

```cpp
    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);
```

改后:

```cpp
    if (i == draft.size()) {
        LOG_DBG("Speculative Simple drafted all past!");
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);
```

## 7. `examples/speculative/speculative.cpp`

### 5.1 新增 acceptance 分布统计数组

原因: 给多 draft sequence 版本也补一份接受长度统计。

改前:

```cpp
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
```

改后:

```cpp
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_seq_accept[10] = {0};

    int n_past_tgt = inp.size();
```

### 5.2 在 draft 命中时累加对应接受长度

改前:

```cpp
                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
```

改后:

```cpp
                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());
                            n_seq_accept[i_dft] += 1;
                            s_keep = s;
                            accept = true;
                        } else {
```

### 5.3 在结束统计里输出 `n_seq_accept`

改前:

```cpp
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
```

改后:

```cpp
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("n_accept  = %d\n", n_accept);
    for (int i = 0; i < 10; i++)
    {
        LOG_INF("n_seq_accept[%d]-accept_len-%d\n", i,n_seq_accept[i]);
        
    }
    
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
```

## 8. `src/llama-context.cpp`

### 6.1 调整 EAGLE3 graph type 自动判定

原因: 把 EAGLE3 的 encoder / decoder 判定逻辑改成优先看 `target_tok_embd` 是否存在。

改前:

```cpp
    // EAGLE3: auto-detect encoder (embeddings+no target_model) or decoder (has target_model)
    llm_graph_type gtype = LLM_GRAPH_TYPE_DEFAULT;
    if (model.arch == LLM_ARCH_EAGLE3) {
        if (cparams.embeddings && model.target_tok_embd == nullptr) {
            gtype = LLM_GRAPH_TYPE_ENCODER;
        } else if (model.target_tok_embd != nullptr) {
            gtype = LLM_GRAPH_TYPE_DECODER;
        }
    }
```

改后:

```cpp
    // EAGLE3: auto-detect encoder (embeddings+no target_model) or decoder (has target_model)
    llm_graph_type gtype = LLM_GRAPH_TYPE_DEFAULT;
    // if (model.arch == LLM_ARCH_EAGLE3) {
    //     if (cparams.embeddings && model.target_tok_embd == nullptr) {
    //         gtype = LLM_GRAPH_TYPE_ENCODER;
    //     } else if (model.target_tok_embd != nullptr) {
    //         gtype = LLM_GRAPH_TYPE_DECODER;
    //     }
    // }
    //
    // 修改后：
    if (model.arch == LLM_ARCH_EAGLE3) {
        if (model.target_tok_embd != nullptr) {
            gtype = LLM_GRAPH_TYPE_DECODER;
        } else {
            gtype = LLM_GRAPH_TYPE_ENCODER;
        }
    }
```

## 9. `src/llama-memory-hybrid.cpp`

### 7.1 `seq_rm()` 从“recurrent 失败就直接返回”改成“两边都清理”

原因: 处理 speculative rollback 时，即使 recurrent cache 部分删除失败，也仍然尝试清理 attention KV，避免位置冲突。

改前:

```cpp
bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}
```

改后:

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

## 10. 文档文件

### 8.1 `MY/speculative_slot_change_record.md`

改前:

```text
文件不存在
```

改后:

```text
新增本文件，集中记录这轮手工修改点及对应改前/改后代码。
```

### 8.2 `MY/speculative_hybrid_fix_changelog.md`

改前:

```text
文件不存在
```

改后:

```text
新增 hybrid / speculative 修复过程的阶段性变更说明。
```

### 8.3 `MY/speculative_decoding_hybrid_comparison.md`

改前:

```text
文件不存在
```

改后:

```text
新增 speculative decoding 在 hybrid 相关路径上的方案对比说明。
```

## 11. 最新收口状态

这份文档前面记录了整个排查过程里出现过的修改。

但到当前这一版，已经按“最小程度修改、不影响原功能”的要求继续收口，真正保留下来的 speculative 相关代码改动只剩:

1. `examples/speculative-simple/speculative-simple.cpp`

已经回退、不再保留生效的修改:

1. `common/sampling.cpp`
2. `src/llama-batch.cpp`
3. `src/llama-kv-cache.cpp`
4. `src/llama-memory-hybrid.cpp`

当前 `speculative-simple.cpp` 的保留改动只有 3 类：

1. 为 target 预留额外 seq slot:
   `params.kv_unified = true;`
   `params.n_parallel = std::max(params.n_parallel, params.speculative.n_max + 2);`
2. 新增本地 helper `speculative_slot_chain`
3. verify 主循环从“一次性 batch verify”改成“逐 token 沿链表/slot 链推进”

### 11.1 当前最小 patch 的核心结构

新增本地 helper:

```cpp
struct speculative_slot_chain {
    struct node {
        llama_seq_id seq_id = -1;
        int prev = -1;
        int depth = 0;
    };

    llama_memory_t mem = nullptr;
    llama_seq_id live_seq = 0;
    int max_slots = 0;
    int tail = -1;
    std::vector<node> nodes;

    void reset();
    llama_seq_id append();
    const node & find_depth(int depth) const;
    void promote_depth(int depth);
};
```

作用:

1. `reset()` 只保留 `seq 0` 作为 live state
2. `append()` 每次基于链尾复制出下一个 slot
3. `find_depth(k)` 可以顺着 `prev` 找到第 `k` 层已接受状态
4. `promote_depth(k)` 直接把该状态 promote 回 `seq 0`

### 11.2 当前 verify 逻辑

不再做一次性多 token verify，而是:

1. `slot_chain.reset()`
2. decode `id_last` 到第一个 slot
3. decode `draft0` 到第二个 slot
4. decode `draft1` 到第三个 slot
5. ...
6. 一旦第 `i` 个 draft reject，链上第 `i` 个已接受状态天然还在
7. `slot_chain.promote_depth(ids.size())` 直接回到接受位置

### 11.3 当前已知问题

这版逻辑上是对的，但性能明显退化，甚至可能比不开 EAGLE 还慢。

直接原因也很明确:

1. 原 speculative verify 倾向于把多个 token 合成一次 target decode
2. 现在这版最小 patch 变成“每个 verify token 单独 decode 一次”
3. 所以 target 端 graph 调度、kernel launch、同步开销被放大了
4. hybrid/recurrent 状态是对了，但吞吐掉下来了

也就是说，这版更像是:

1. 先用最小 patch 验证“状态链条正确”
2. 再在这个正确版本上讨论怎么把多个步骤重新合并、把性能拿回来
