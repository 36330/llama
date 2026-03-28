# Qwen3.5 EAGLE3 推测解码支持 - 完整实现计划

## 1. 背景与目标

### 1.1 什么是 EAGLE3

EAGLE3 是一种推测解码（Speculative Decoding）方法，通过一个轻量级的 draft 模型预测多个未来 token，然后由目标模型一次性验证，从而加速推理。

EAGLE3 的核心架构：
- **Encoder**（`llm_build_eagle3_encode`）：从目标模型的 3 个中间层提取特征（low/mid/high），通过 FC 层融合为 `g_embeddings`
- **Decoder**（`llm_build_eagle3_decode`）：单层 Transformer，接收 token embeddings + g_embeddings，输出 draft logits

### 1.2 当前状态

| 模型 | EAGLE3 中间层提取 | 说明 |
|------|:-:|------|
| Qwen3 (`qwen3.cpp`) | **已实现** | 完整支持 |
| Qwen3VL-MoE (`qwen3vl-moe.cpp`) | **已实现** | 完整支持 |
| Qwen3MoE (`qwen3moe.cpp`) | **已实现** | 完整支持 |
| **Qwen3.5** (`qwen35.cpp`) | **未实现** | 缺少中间层提取 |
| **Qwen3.5-MoE** (`qwen35moe.cpp`) | **未实现** | 缺少中间层提取 |

### 1.3 目标

为 Qwen3.5 和 Qwen3.5-MoE 添加完整的 EAGLE3 推测解码支持。

---

## 2. 架构差异分析

### 2.1 Qwen3 vs Qwen3.5 的关键差异

Qwen3.5 相比 Qwen3 有一个**根本性的架构变化**：混合注意力机制。

| 特性 | Qwen3 | Qwen3.5 |
|------|-------|---------|
| 注意力类型 | 全部是标准 Self-Attention | **混合**: Full Attention + Linear Attention (Delta Net) |
| 基类 | `llm_graph_context` | `llm_build_delta_net_base` |
| 记忆机制 | KV Cache | **Hybrid**: KV Cache + Recurrent State (SSM) |
| FFN | 标准 SwiGLU | 标准 SwiGLU（Dense）/ MoE + Shared Expert（MoE版） |
| Post-Attn Norm | 无 | **有** `attn_post_norm` |
| Q 投影 | 独立 Q | **Joint QG 投影**（Query + Gate 合并） |
| 注意力门控 | 无 | **Sigmoid Gate** 门控注意力 |
| 层类型判断 | 无 | `hparams.is_recurrent(il)` 区分层类型 |

### 2.2 这些差异对 EAGLE3 的影响

EAGLE3 中间层提取的核心逻辑是：**在目标模型的 for 循环中，指定层的输入（`inpL`）被标记为提取点**。

```cpp
// 这段代码需要插入到 qwen35.cpp 和 qwen35moe.cpp 的 for 循环开头
if (eagle3 && cparams.eagle3_extract_enabled && !eagle3->extract_layer_indices.empty()) {
    static const char * eagle3_extract_names[] = {"eagle3_extract_0", "eagle3_extract_1", "eagle3_extract_2"};
    for (size_t i = 0; i < eagle3->extract_layer_indices.size() && i < 3; ++i) {
        if (eagle3->extract_layer_indices[i] == il) {
            cb(inpL, eagle3_extract_names[i], il);
            break;
        }
    }
}
```

**关键问题**：Qwen3.5 的混合注意力层中，线性注意力层（Delta Net）使用 recurrent state 而非 KV Cache。EAGLE3 提取的是**层输入的隐藏状态**（`inpL`），这与注意力类型无关——无论是 Full Attention 层还是 Linear Attention 层，`inpL` 都是上一层的输出隐藏状态，维度一致（`[n_embd, n_tokens]`）。因此**EAGLE3 的中间层提取逻辑对混合注意力架构透明**。

### 2.3 EAGLE3 Encoder/Decoder 是否需要修改

**不需要。** EAGLE3 的 Encoder 和 Decoder 是与目标模型架构无关的独立组件：
- Encoder 只关心输入的 `3 * target_hidden_size` 维特征
- Decoder 是标准的单层 Transformer（使用 RoPE + KV Cache）

只要目标模型正确提取了 3 个中间层特征，EAGLE3 就能工作。

---

## 3. 需要修改的文件

### 3.1 文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/models/qwen35.cpp` | **代码修改** | 添加 EAGLE3 中间层提取（核心） |
| `src/models/qwen35moe.cpp` | **代码修改** | 添加 EAGLE3 中间层提取（核心） |
| `convert_hf_to_gguf.py` | **代码修改** | 添加 Qwen3.5 EAGLE3 draft 模型识别与转换 |

### 3.2 可能不需要修改的文件

| 文件 | 原因 |
|------|------|
| `src/models/eagle3.cpp` | Encoder/Decoder 与目标模型无关，无需修改 |
| `src/llama-arch.h` | EAGLE3 架构已注册，Qwen3.5 架构已注册 |
| `src/llama-arch.cpp` | 张量映射已完整 |
| `gguf-py/gguf/constants.py` | EAGLE3 和 Qwen3.5 常量已完整 |
| `src/llama-model.cpp` | EAGLE3 模型加载逻辑已完整 |

---

## 4. 详细实现步骤

### 步骤 1：修改 `src/models/qwen35.cpp` — 添加 EAGLE3 中间层提取

**位置**：`llm_build_qwen35` 构造函数中，`for (int il = 0; il < n_layer; ++il)` 循环内部，在 `ggml_tensor * inpSA = inpL;` 之后、`cur = build_norm(...)` 之前。

**参考**：`src/models/qwen3.cpp` 第 24-33 行

**修改前**（qwen35.cpp 第 26-29 行）：
```cpp
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;

    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
```

**修改后**：
```cpp
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;

    // EAGLE3: Extract intermediate layer features from target model at layer INPUT
    if (eagle3 && cparams.eagle3_extract_enabled && !eagle3->extract_layer_indices.empty()) {
        static const char * eagle3_extract_names[] = {"eagle3_extract_0", "eagle3_extract_1", "eagle3_extract_2"};
        for (size_t i = 0; i < eagle3->extract_layer_indices.size() && i < 3; ++i) {
            if (eagle3->extract_layer_indices[i] == il) {
                cb(inpL, eagle3_extract_names[i], il);
                break;
            }
        }
    }

    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
```

### 步骤 2：修改 `src/models/qwen35moe.cpp` — 添加 EAGLE3 中间层提取

**位置**：与 qwen35.cpp 完全相同的位置。

**修改前**（qwen35moe.cpp 第 26-29 行）：
```cpp
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;

    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
```

**修改后**：
```cpp
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;

    // EAGLE3: Extract intermediate layer features from target model at layer INPUT
    if (eagle3 && cparams.eagle3_extract_enabled && !eagle3->extract_layer_indices.empty()) {
        static const char * eagle3_extract_names[] = {"eagle3_extract_0", "eagle3_extract_1", "eagle3_extract_2"};
        for (size_t i = 0; i < eagle3->extract_layer_indices.size() && i < 3; ++i) {
            if (eagle3->extract_layer_indices[i] == il) {
                cb(inpL, eagle3_extract_names[i], il);
                break;
            }
        }
    }

    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
```

### 步骤 3：修改 `convert_hf_to_gguf.py` — 添加 Qwen3.5 EAGLE3 模型转换支持

这是最复杂的部分。需要让转换脚本能识别 Qwen3.5 的 EAGLE3 draft 模型。

#### 3a. 问题分析

当前 EAGLE3 的检测逻辑在 `LlamaModel` 类的 `__init__` 中（第 2769 行）：
```python
if "draft_vocab_size" in self.hparams and self.hparams["num_hidden_layers"] == 1:
    self.is_eagle3 = True
    self.model_arch = gguf.MODEL_ARCH.EAGLE3
```

这段检测逻辑只会在 `LlamaModel` 类中触发（由 `@ModelBase.register("LlamaForCausalLM", ...)` 注册）。

**Qwen3.5 的 EAGLE3 draft 模型**会使用不同的 HuggingFace 架构名称（如 `Eagle3Qwen3_5ForCausalLM` 或类似命名），需要：
1. 在 `Qwen3_5TextModel` 或新的注册类中添加 EAGLE3 检测
2. 或者将 EAGLE3 的识别名称添加到 Qwen3.5 的 `@ModelBase.register` 中

#### 3b. 方案选择

**推荐方案**：在 `Qwen3_5TextModel` 类中添加 EAGLE3 检测逻辑（参考 LlamaModel 的实现）。

具体取决于 Qwen3.5 EAGLE3 draft 模型的 `config.json` 中的 `architectures` 字段名称。可能的命名：
- `Eagle3Qwen3_5ForCausalLM`
- `Qwen3_5ForCausalLMEagle3`
- 或者直接使用 `Qwen3_5ForCausalLM`（通过 `draft_vocab_size` 字段区分）

#### 3c. 实现思路

**方案 A**（如果 draft 模型的 architecture 名称与基础模型相同，靠 `draft_vocab_size` 区分）：

在 `Qwen3_5TextModel` 类中添加 `__init__` 方法：

```python
@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM",
                     "Eagle3Qwen3_5ForCausalLM", "Qwen3_5ForCausalLMEagle3")
class Qwen3_5TextModel(_LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # detect EAGLE-3 Qwen3.5 checkpoint
        if "draft_vocab_size" in self.hparams and self.hparams["num_hidden_layers"] == 1:
            self.is_eagle3 = True
            self.model_arch = gguf.MODEL_ARCH.EAGLE3
            logger.info("Detected EAGLE-3 Qwen3.5 draft model, switching to EAGLE3 architecture")
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_architecture()
            # ... (与 LlamaModel 中相同的 EAGLE3 参数读取逻辑)
```

**方案 B**（如果 draft 模型使用独立的 architecture 名称）：

创建独立的注册类，继承 `Qwen3_5TextModel` 并覆盖 `__init__`。

**注意**：具体方案需要根据实际训练出来的 Qwen3.5 EAGLE3 draft 模型的 `config.json` 格式来确定。目前社区还没有公开的 Qwen3.5 EAGLE3 模型，这部分需要等模型训练完成后再做适配。

---

## 5. EAGLE3 工作流全览

```
                         目标模型 (Qwen3.5)
                              │
                   ┌──────────┼──────────┐
                   │          │          │
              Layer[2]   Layer[N/2]  Layer[N-3]
              (low)      (mid)       (high)
                   │          │          │
                   └──────────┼──────────┘
                              │
                     ┌────────▼────────┐
                     │   3 * hidden    │  3个中间层特征拼接
                     │  [3*4096, T]    │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  EAGLE3 Encoder │  FC 融合层
                     │  fc: 3H → H    │
                     └────────┬────────┘
                              │
                      g_embeddings [4096, T]
                              │
                     ┌────────▼────────┐
                     │  EAGLE3 Decoder │  单层 Transformer
                     │  (1 layer)      │
                     │  input: tok_embd│
                     │       + g_embd  │
                     └────────┬────────┘
                              │
                      draft logits → draft tokens
                              │
                     ┌────────▼────────┐
                     │  目标模型验证    │  一次前向传播验证多个 token
                     └─────────────────┘
```

### 5.1 提取层索引

提取层的默认计算规则（`convert_hf_to_gguf.py` 第 2793 行）：
```python
extract_layers = [2, target_num_layers // 2, target_num_layers - 3]
```

例如 Qwen3.5-4B（36 层）：`extract_layers = [2, 18, 33]`
例如 Qwen3.5-72B-MoE（如果有 64 层）：`extract_layers = [2, 32, 61]`

### 5.2 运行时数据流

1. 目标模型前向传播时，在 `extract_layers` 指定的层，将 `inpL`（层输入隐藏状态）通过 `cb()` 回调标记
2. 框架收集 3 个提取点的特征，拼接为 `[3 * hidden_size, n_tokens]`
3. EAGLE3 Encoder 通过 FC 层将其压缩为 `[hidden_size, n_tokens]` 的 `g_embeddings`
4. EAGLE3 Decoder 接收 `g_embeddings` + token embeddings，通过单层 Transformer 生成 draft logits
5. Draft logits 生成候选 token 序列
6. 目标模型一次性验证候选序列

---

## 6. 关于 Qwen3.5 混合注意力的特殊考虑

### 6.1 Linear Attention 层的特殊性

Qwen3.5 的 Linear Attention 层使用 **Gated Delta Net**，这是一种 recurrent 机制：
- 使用 SSM（State Space Model）的 convolution + delta rule
- 维护 recurrent state（conv_states + ssm_states）
- 不使用 KV Cache

**EAGLE3 的中间层提取与此无关**。提取的是层输入的隐藏状态 `inpL`，这在所有层类型中都是一致的 `[n_embd, n_tokens]` 张量。

### 6.2 提取层选择建议

当某些层是 Linear Attention 层时，提取的隐藏状态仍然是有效的表示。但如果想要最佳效果，可以考虑：
- 确保提取层涵盖不同类型的层（既有 Full Attention 层也有 Linear Attention 层）
- 默认规则 `[2, N/2, N-3]` 通常已经能覆盖两种层类型

### 6.3 EAGLE3 Decoder 与 Qwen3.5 的不兼容点

EAGLE3 Decoder 使用标准的 Self-Attention + KV Cache（在 `eagle3.cpp` 中实现），**不使用 Linear Attention**。这是正确的设计：
- Draft 模型只有 1 层，不需要 recurrent 机制
- Draft 模型的目的是快速预测，标准 attention 足够

---

## 7. EAGLE3 Draft 模型训练（参考）

### 7.1 训练数据准备

EAGLE3 draft 模型需要以下训练数据：
- 目标模型在训练集上的中间层特征（3 个提取层）
- 对应的 token 序列

### 7.2 模型结构

Draft 模型的 `config.json` 应包含：
```json
{
    "architectures": ["Eagle3Qwen3_5ForCausalLM"],
    "num_hidden_layers": 1,
    "hidden_size": 2560,          // 与目标模型相同
    "num_attention_heads": 20,    // 与目标模型相同
    "num_key_value_heads": 4,     // 与目标模型相同
    "intermediate_size": 6912,    // 与目标模型相同
    "draft_vocab_size": 151936,   // 词表大小
    "target_hidden_size": 2560,   // 目标模型隐藏维度
    "vocab_size": 151936
}
```

### 7.3 模型权重结构

```
model.fc.weight           # [hidden_size, 3 * target_hidden_size] — 特征融合层
model.layers.0.input_layernorm.weight      # [hidden_size] — 输入规范化
model.layers.0.eagle3_hidden_norm.weight   # [hidden_size] — 隐藏状态规范化
model.layers.0.self_attn.q_proj.weight     # [n_heads * head_dim, 2 * hidden_size] — Q投影
model.layers.0.self_attn.k_proj.weight     # [n_kv_heads * head_dim, 2 * hidden_size]
model.layers.0.self_attn.v_proj.weight     # [n_kv_heads * head_dim, 2 * hidden_size]
model.layers.0.self_attn.o_proj.weight     # [hidden_size, n_heads * head_dim]
model.layers.0.post_attention_layernorm.weight # [hidden_size]
model.layers.0.mlp.gate_proj.weight        # [intermediate_size, hidden_size]
model.layers.0.mlp.up_proj.weight          # [intermediate_size, hidden_size]
model.layers.0.mlp.down_proj.weight        # [hidden_size, intermediate_size]
model.norm.weight                          # [hidden_size] — 输出规范化
lm_head.weight                             # [vocab_size, hidden_size] — 词汇投影
```

---

## 8. GGUF 转换命令（预期）

当 Qwen3.5 EAGLE3 draft 模型可用时，转换命令为：

```bash
# 转换 draft 模型
python convert_hf_to_gguf.py \
    /path/to/eagle3-qwen35-draft \
    --target-model-dir /path/to/Qwen3.5-4B \
    --outfile eagle3-qwen35-4b-draft.gguf

# 量化（可选）
./build/bin/llama-quantize eagle3-qwen35-4b-draft.gguf eagle3-qwen35-4b-draft-Q8_0.gguf Q8_0
```

### 8.1 推理命令（预期）

```bash
./build/bin/llama-speculative \
    -m /path/to/Qwen3.5-4B.gguf \
    -md eagle3-qwen35-4b-draft.gguf \
    --draft-max 4 \
    -p "Hello, how are you?"
```

---

## 9. 实施优先级与时间线

### Phase 1：目标模型侧修改（可立即执行）
1. **修改 `qwen35.cpp`**：添加 EAGLE3 提取代码（5 分钟）
2. **修改 `qwen35moe.cpp`**：添加 EAGLE3 提取代码（5 分钟）
3. **编译验证**：确保编译通过（10 分钟）

### Phase 2：转换脚本适配（需要 draft 模型 config 格式确定后）
4. **修改 `convert_hf_to_gguf.py`**：添加 Qwen3.5 EAGLE3 识别逻辑
5. **测试转换流程**

### Phase 3：端到端测试（需要训练好的 draft 模型）
6. **训练 Qwen3.5 EAGLE3 draft 模型**
7. **GGUF 转换测试**
8. **推理性能测试**

---

## 10. 参考代码位置

| 参考内容 | 文件 | 行号 |
|---------|------|------|
| Qwen3 EAGLE3 提取（参考模板） | `src/models/qwen3.cpp` | 24-33 |
| Qwen3VL-MoE EAGLE3 提取 | `src/models/qwen3vl-moe.cpp` | 30-39 |
| EAGLE3 Encoder 实现 | `src/models/eagle3.cpp` | 3-38 |
| EAGLE3 Decoder 实现 | `src/models/eagle3.cpp` | 43-187 |
| EAGLE3 模型加载 | `src/llama-model.cpp` | 2349-2373, 6924-6978 |
| EAGLE3 图构建入口 | `src/llama-model.cpp` | 8712-8717 |
| LlamaModel EAGLE3 检测 | `convert_hf_to_gguf.py` | 2768-2809 |
| Qwen3.5 转换类 | `convert_hf_to_gguf.py` | 5195-5197 |
| EAGLE3 GGUF 常量 | `gguf-py/gguf/constants.py` | 3553-3570 |
| EAGLE3 架构定义 | `src/llama-arch.h` | 137, 551-553 |
| Qwen3.5 架构定义 | `src/llama-arch.h` | 1037-1062 |
| Qwen3.5 类声明 | `src/models/models.h` | 580-581, 614-615 |

---

## 11. 风险与注意事项

1. **混合注意力的 extract_layers 选择**：默认的 `[2, N/2, N-3]` 可能落在 Linear Attention 层上。理论上不影响功能，但最优提取层可能需要实验调优。

2. **Qwen3.5 的 Post-Attention Norm**：Qwen3.5 在 attention 后有额外的 `attn_post_norm`，但这不影响 EAGLE3 提取（提取的是层输入 `inpL`，在 norm 之前）。

3. **Draft 模型的 RoPE 类型**：Qwen3.5 使用 MRoPE（多维旋转位置编码，带 sections），而 EAGLE3 Decoder 使用标准 RoPE。这需要确认 draft 模型训练时使用的 RoPE 类型。如果 draft 模型使用标准 RoPE（当前 `eagle3.cpp` 中的 `ggml_rope_ext`），则无需修改。

4. **词表兼容性**：Qwen3.5 的词表 (`vocab_size=151936`) 与 EAGLE3 draft 模型的 `draft_vocab_size` 需要一致，或者使用 `d2t` 映射张量进行转换。

5. **目前没有公开的 Qwen3.5 EAGLE3 draft 模型**：Phase 1 可以先完成，Phase 2/3 需要等训练好的模型。
