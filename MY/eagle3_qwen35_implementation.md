# EAGLE3 新模型适配完整技术文档

以 Qwen3.5 适配为实例，记录从代码修改到测试验证的全流程。适配其他模型时按本文档逐步操作即可。

---

## 1. 整体流程概览

```
Step 1: 修改 C++ 代码（Target 模型添加中间层提取）
  ↓  验证：编译通过 + Target 模型单独推理不受影响
Step 2: 转换 Draft 模型为 GGUF
  ↓  验证：转换成功 + GGUF 元数据正确
Step 3: 联动测试（Target + Draft 联合推理）
  ↓  验证：推测解码正常输出 + 加速比合理
```

**本次 Qwen3.5 适配涉及的文件：**

| 文件 | 改动类型 | 改了什么 |
|------|---------|---------|
| `src/models/qwen35.cpp` | 新增 11 行 | 添加 EAGLE3 中间层提取代码 |
| `src/models/qwen35moe.cpp` | 新增 11 行 | 添加 EAGLE3 中间层提取代码（与上面完全相同） |
| `convert_hf_to_gguf.py` | **无改动** | 已有的 LlamaModel EAGLE3 检测逻辑可直接复用 |
| `src/models/eagle3.cpp` | **无改动** | EAGLE3 Encoder/Decoder 与 target 架构无关 |

---

## 2. Step 1：修改 Target 模型 C++ 代码

### 2.1 确定需要修改的文件

```bash
# 查看目标模型的源文件
ls src/models/qwen35*.cpp
# 输出：
# src/models/qwen35.cpp       ← Dense 版本
# src/models/qwen35moe.cpp    ← MoE 版本

# 确认目前没有 EAGLE3 提取代码
grep "eagle3_extract" src/models/qwen35.cpp
# 输出为空 → 需要添加
```

**规则：如果模型有 Dense 和 MoE 两个变体，两个文件都要改。**

### 2.2 定位插入位置

在每个文件中找到主 for 循环的开头：

```bash
grep -n "for.*il.*n_layer" src/models/qwen35.cpp
# 输出：26:    for (int il = 0; il < n_layer; ++il) {

grep -n "inpSA = inpL" src/models/qwen35.cpp
# 输出：27:        ggml_tensor * inpSA = inpL;

grep -n "build_norm.*attn_norm" src/models/qwen35.cpp
# 输出：29:        cur = build_norm(inpL, model.layers[il].attn_norm, ...
```

**插入位置 = 第 27 行（`inpSA = inpL`）之后、第 29 行（`build_norm`）之前。**

可以参考已有实现确认模式：

```bash
# 参考 Qwen3 的实现（已验证可工作）
sed -n '21,40p' src/models/qwen3.cpp
```

### 2.3 具体修改内容

#### 文件 1：`src/models/qwen35.cpp`

**修改前**（第 26-29 行）：

```cpp
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
```

**修改后**（第 26-40 行）：

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

#### 文件 2：`src/models/qwen35moe.cpp`

**完全相同的修改，相同的位置（第 26-29 行 → 第 26-40 行）。**

### 2.4 完整 diff

```diff
diff --git a/src/models/qwen35.cpp b/src/models/qwen35.cpp
--- a/src/models/qwen35.cpp
+++ b/src/models/qwen35.cpp
@@ -26,6 +26,17 @@
     for (int il = 0; il < n_layer; ++il) {
         ggml_tensor * inpSA = inpL;

+        // EAGLE3: Extract intermediate layer features from target model at layer INPUT
+        if (eagle3 && cparams.eagle3_extract_enabled && !eagle3->extract_layer_indices.empty()) {
+                static const char * eagle3_extract_names[] = {"eagle3_extract_0", "eagle3_extract_1", "eagle3_extract_2"};
+                for (size_t i = 0; i < eagle3->extract_layer_indices.size() && i < 3; ++i) {
+                    if (eagle3->extract_layer_indices[i] == il) {
+                        cb(inpL, eagle3_extract_names[i], il);
+                        break;
+                    }
+                }
+            }
+
         cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
```

`qwen35moe.cpp` 的 diff 完全一致。

### 2.5 代码解释

这段代码做了什么：

```
eagle3                          → 指向 EAGLE3 上下文的指针，非 EAGLE3 模式时为 nullptr
cparams.eagle3_extract_enabled  → 是否启用了 EAGLE3 提取（由推理参数控制）
eagle3->extract_layer_indices   → 需要提取的层索引列表，如 [2, 16, 29]
cb(inpL, "eagle3_extract_0", il)→ 将当前层的输入隐藏状态标记为提取点
```

当 EAGLE3 未启用时（普通推理），`eagle3` 为 nullptr，整个 if 块不执行，零性能开销。

### 2.6 为什么同一段代码适用于所有模型

提取的是 `inpL`（层输入隐藏状态），维度为 `[n_embd, n_tokens]`，与具体层内部的注意力类型无关：
- Full Attention 层 → inpL 是上一层输出，`[n_embd, n_tokens]`
- Linear Attention / Delta Net 层 → inpL 同样是上一层输出，`[n_embd, n_tokens]`
- MoE FFN 层 → inpL 同上

### 2.7 Step 1 单独验证

#### 验证 1：编译通过

```bash
rm -rf build && mkdir build && cd build
cmake .. -G Ninja \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS=gfx1151 \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DGGML_CUDA_GRAPHS=OFF \
    -DGGML_VULKAN=ON
ninja
```

**预期结果**：编译无错误无警告。如果编译报错，多半是插入位置不对或缺少花括号。

#### 验证 2：Target 模型单独推理不受影响

修改后的代码在非 EAGLE3 模式下应该完全透明。验证方式：

```bash
# 用修改后的 binary 运行 target 模型的普通推理
./build/bin/llama-cli \
    -m models/Qwen3.5-4B-BF16.gguf \
    -p "Hello, how are you?" \
    -n 50 -ngl 99
```

**预期结果**：正常输出文本，与修改前完全一致。如果崩溃或输出异常，说明代码插入位置有误。

#### 验证 3：确认提取代码存在

```bash
grep -c "eagle3_extract" src/models/qwen35.cpp
# 预期输出：3（eagle3_extract_0, eagle3_extract_1, eagle3_extract_2 各出现一次在数组定义中）

grep -c "eagle3_extract" src/models/qwen35moe.cpp
# 预期输出：3
```

---

## 3. Step 2：转换 Draft 模型为 GGUF

### 3.1 前提检查

#### 检查 draft 模型目录内容

```bash
ls <draft_model_dir>/
# 必须包含：
#   config.json
#   model.safetensors (或 model-00001-of-XXXXX.safetensors)
```

#### 检查 config.json 关键字段

```bash
python3 -c "
import json
with open('<draft_model_dir>/config.json') as f:
    c = json.load(f)
print('architectures:', c.get('architectures'))
print('num_hidden_layers:', c.get('num_hidden_layers'))
print('draft_vocab_size:', c.get('draft_vocab_size'))
print('hidden_size:', c.get('hidden_size'))
print('vocab_size:', c.get('vocab_size'))
"
```

**必须满足的条件：**
- `architectures` 包含已注册的名称（`LlamaForCausalLMEagle3` 等）
- `num_hidden_layers` == 1
- `draft_vocab_size` 存在
- `hidden_size` 与 target 模型一致

#### 检查 draft 模型 tensor 结构

```bash
python3 -c "
from safetensors import safe_open
with safe_open('<draft_model_dir>/model.safetensors', framework='pt') as f:
    for k in f.keys():
        t = f.get_tensor(k)
        print(f'{k}: {list(t.shape)} {t.dtype}')
"
```

**本次 Qwen3.5-4B 的实际 tensor 结构：**

```
d2t: [32000] torch.int64                         ← Draft→Target 词表映射
fc.weight: [2560, 7680] torch.bfloat16            ← 3 路特征融合 (7680 = 3 * 2560)
lm_head.weight: [32000, 2560] torch.bfloat16      ← 词汇投影头
midlayer.hidden_norm.weight: [2560] torch.bfloat16
midlayer.input_layernorm.weight: [2560] torch.bfloat16
midlayer.mlp.down_proj.weight: [2560, 9216] torch.bfloat16
midlayer.mlp.gate_proj.weight: [9216, 2560] torch.bfloat16
midlayer.mlp.up_proj.weight: [9216, 2560] torch.bfloat16
midlayer.post_attention_layernorm.weight: [2560] torch.bfloat16
midlayer.self_attn.k_proj.weight: [1024, 5120] torch.bfloat16  ← 输入 5120 = 2 * 2560
midlayer.self_attn.o_proj.weight: [2560, 4096] torch.bfloat16
midlayer.self_attn.q_proj.weight: [4096, 5120] torch.bfloat16  ← 输入 5120 = 2 * 2560
midlayer.self_attn.v_proj.weight: [1024, 5120] torch.bfloat16
norm.weight: [2560] torch.bfloat16
t2d: [248320] torch.bool                          ← Target→Draft 映射（转换时跳过）
```

**校验要点：**
- `fc.weight` 第二维 = 3 * hidden_size
- Q/K/V 的输入维度 = 2 * hidden_size（因为拼接了 token embedding + g_embedding）
- `lm_head.weight` 第一维 = draft_vocab_size
- `d2t` 长度 = draft_vocab_size

#### 检查 target HF 目录

```bash
# target HF 目录必须包含 config.json 和 tokenizer 文件
ls <target_hf_dir>/config.json <target_hf_dir>/tokenizer.json <target_hf_dir>/tokenizer_config.json
```

如果 target 的 config.json 是 VL 模型格式（包含 `text_config` 子字段），转换脚本会自动读取 `text_config`。

### 3.2 执行转换

```bash
python convert_hf_to_gguf.py \
    <draft_model_dir> \
    --target-model-dir <target_hf_dir> \
    --outfile <output_path>.gguf
```

**本次实际执行的命令：**

```bash
python convert_hf_to_gguf.py \
    /home/ljl/ljl_test/modelss/qwen3.5-9b-eagle \
    --target-model-dir /home/ljl/ljl_test/llama/models/qwen3.5-9b \
    --outfile models/eagle3-qwen3.5-9b-eagle.gguf
```

### 3.3 Step 2 单独验证

#### 验证 1：转换日志关键行

转换过程中检查以下日志输出：

```
INFO: Detected EAGLE-3 draft model, switching to EAGLE3 architecture
INFO: EAGLE3: extract_layers = [2, 16, 29] (target model has 32 layers)
INFO: EAGLE3: target_hidden_size = 2560 (from target model config)
INFO: Model successfully exported to models/eagle3-qwen35-4b-draft-f16.gguf
```

**逐行确认：**
- `switching to EAGLE3 architecture` → EAGLE3 自动检测触发
- `extract_layers = [2, 16, 29]` → 提取层索引正确（公式：`[2, N//2, N-3]`，N=32）
- `target_hidden_size = 2560` → 与 target 模型 hidden_size 一致
- `successfully exported` → 转换成功

#### 验证 2：检查 GGUF 文件大小

```bash
ls -lh models/eagle3-qwen35-4b-draft-f16.gguf
# 预期：约 420MB（BF16 精度）
# 如果文件异常小（< 10MB）或为 0，说明转换有问题
```

#### 验证 3：检查 GGUF 元数据

```bash
# 用 gguf-dump 查看关键元数据
python3 -c "
from gguf import GGUFReader
reader = GGUFReader('models/eagle3-qwen35-4b-draft-f16.gguf')
for field in reader.fields.values():
    if 'eagle3' in field.name or 'arch' in field.name or 'extract' in field.name:
        print(f'{field.name} = {field.parts[-1]}')
# 查看 tensor 列表
for tensor in reader.tensors:
    print(f'{tensor.name}: shape={tensor.shape}')
"
```

**预期输出应包含：**
- `general.architecture = eagle3`
- `eagle3.extract_layers = [2, 16, 29]`
- `eagle3.target_hidden_size = 2560`
- tensor 列表中有 `fc.weight`、`blk.0.attn_q.weight` 等

#### 验证 4：如果转换失败的排查

| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| `EAGLE3 model requires --target-model-dir` | 缺少 target 目录参数 | 添加 `--target-model-dir` |
| `KeyError: 'num_hidden_layers'` | target config.json 格式不对 | 检查是否有 `text_config` 嵌套 |
| `No module named 'safetensors'` | Python 环境缺少依赖 | `pip install safetensors torch` |
| Tensor shape 不匹配 | draft 模型训练参数与 config.json 不一致 | 检查 config.json 中的维度参数 |

---

## 4. Step 3：联动测试

### 4.1 基本推理测试

```bash
for prompt in \
    "Write a quicksort algorithm in Python. Write code only." \
    "Explain the Pythagorean theorem" \
    "Plan a 1 day trip to DC"; do
  echo "=== Prompt: $prompt ==="
  ./build/bin/llama-speculative-simple \
      -m <target_gguf> \
      -md <draft_gguf> \
      --eagle3 -p "$prompt" -n 256 --draft 1 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99
done
```

**本次实际执行的命令：**

```bash
for prompt in \
    "Write a quicksort algorithm in Python. Write code only." \
    "Explain the Pythagorean theorem" \
    "Plan a 1 day trip to DC"; do
  echo "=== Prompt: $prompt ==="
  ./build/bin/llama-speculative-simple \
      -m models/Qwen3.5-4B-BF16.gguf \
      -md models/eagle3-qwen35-4b-draft-f16.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 1 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99
done
```

### 4.2 联动测试验证标准

| 检查项 | 正常表现 | 异常表现及排查 |
|--------|---------|--------------|
| 程序启动 | 正常加载两个模型，无 crash | 如果 crash，检查 GGUF 转换是否正确 |
| 输出内容 | 语义连贯的文本 | 乱码/重复 → draft 模型质量问题或 tensor 映射错误 |
| 加速比 | draft acceptance rate > 0（日志中可见） | 0% acceptance → extract_layers 或 hidden_size 不匹配 |
| 与普通推理对比 | 相同 seed + temp 0 下输出应一致 | 输出不同 → EAGLE3 提取代码位置可能有误 |

### 4.3 对比测试：验证输出一致性

```bash
# 1. 普通推理（无 EAGLE3）
./build/bin/llama-cli \
    -m models/Qwen3.5-4B-BF16.gguf \
    -p "What is 2+2?" -n 50 \
    --temp 0 --top-k 1 --seed 42 -ngl 99

# 2. EAGLE3 推理
./build/bin/llama-speculative-simple \
    -m models/Qwen3.5-4B-BF16.gguf \
    -md models/eagle3-qwen35-4b-draft-f16.gguf \
    --eagle3 -p "What is 2+2?" -n 50 --draft 1 \
    --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99
```

**预期：两者输出内容完全一致**（因为推测解码只加速，不改变输出分布）。

### 4.4 如果联动测试失败的排查流程

```
联动测试失败
  ├─ 启动就 crash
  │    ├─ "failed to load model" → GGUF 文件损坏，重新转换
  │    ├─ "tensor shape mismatch" → draft 和 target 的 hidden_size 不匹配
  │    └─ segfault → C++ 提取代码位置有误，review diff
  │
  ├─ 输出乱码
  │    ├─ draft 模型训练质量差 → 换 checkpoint 或检查训练
  │    └─ d2t 词表映射错误 → 检查 draft config 的 draft_vocab_size 和 vocab_size
  │
  └─ 加速比为 0 或负数
       ├─ extract_layers 与训练时不一致 → 检查 draft 训练时用的提取层
       └─ target hidden_size 不匹配 → 检查 fc.weight shape 是否为 [hidden_size, 3*hidden_size]
```

---

## 5. 适配其他新模型的通用 Checklist

假设你要为 `ModelX` 添加 EAGLE3 支持：

### Step 0：前置检查（1 分钟）

```bash
# 检查 ModelX 是否已经有 EAGLE3 提取
grep -rn "eagle3_extract" src/models/modelx*.cpp
# 如果有输出 → 不需要改 C++，直接跳到 Step 2

# 查看已支持 EAGLE3 的所有模型（作为参考）
grep -rn "eagle3_extract" src/models/ | grep "_0\""
# 输出类似：
# src/models/llama.cpp:25:    cb(inpL, eagle3_extract_names[i], il);
# src/models/qwen3.cpp:29:    cb(inpL, eagle3_extract_names[i], il);
# src/models/qwen35.cpp:34:   cb(inpL, eagle3_extract_names[i], il);
# ...
```

### Step 1：修改 C++ 文件（5 分钟）

```bash
# 1. 找到模型源文件
ls src/models/modelx*.cpp

# 2. 找插入位置
grep -n "for.*il.*n_layer" src/models/modelx.cpp
grep -n "inpSA = inpL" src/models/modelx.cpp
grep -n "build_norm.*attn_norm" src/models/modelx.cpp

# 3. 复制参考代码，在 "inpSA = inpL" 之后、第一个 "build_norm" 之前插入
#    代码块完全相同，直接从 qwen3.cpp 或本文档复制即可

# 4. 如果有 MoE 变体，同样修改 modelxmoe.cpp
```

**要插入的代码（所有模型通用，一字不改）：**

```cpp
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
```

**单独验证：**
```bash
# 编译
ninja -C build

# 普通推理不受影响
./build/bin/llama-cli -m <target_gguf> -p "Hello" -n 20 -ngl 99
```

### Step 2：转换 Draft 模型（5 分钟）

```bash
# 1. 检查 draft config.json
cat <draft_dir>/config.json | python3 -m json.tool | grep -E "architectures|num_hidden_layers|draft_vocab_size|hidden_size"

# 2. 检查 target HF 目录有 config.json + tokenizer
ls <target_hf_dir>/config.json <target_hf_dir>/tokenizer.json

# 3. 执行转换
python convert_hf_to_gguf.py <draft_dir> --target-model-dir <target_hf_dir> --outfile <output>.gguf

# 4. 确认日志中有：
#    "Detected EAGLE-3 draft model"
#    "extract_layers = [...]"
#    "successfully exported"
```

### Step 3：联动测试（5 分钟）

```bash
# 1. EAGLE3 推理
./build/bin/llama-speculative-simple \
    -m <target_gguf> -md <draft_gguf> --eagle3 \
    -p "Write hello world in Python" -n 128 --draft 1 \
    --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99

# 2. 对比普通推理输出是否一致
./build/bin/llama-cli -m <target_gguf> \
    -p "Write hello world in Python" -n 128 \
    --temp 0 --top-k 1 --seed 42 -ngl 99
```

---

## 6. 不需要改动的部分（重要）

适配新模型时，以下文件/组件是通用的，**不需要修改**：

| 文件 | 说明 |
|------|------|
| `src/models/eagle3.cpp` | EAGLE3 Encoder（FC 融合层）和 Decoder（单层 Transformer），与 target 架构完全无关 |
| `src/llama-arch.h` / `.cpp` | EAGLE3 架构定义和张量名映射，已完整 |
| `src/llama-model.cpp` | EAGLE3 模型加载逻辑和图构建入口，已完整 |
| `gguf-py/gguf/constants.py` | EAGLE3 相关常量，已完整 |
| `convert_hf_to_gguf.py` | 只要 draft 用已注册架构名（`LlamaForCausalLMEagle3` 等），转换逻辑通用 |

**唯一需要改的就是 `src/models/<target_model>.cpp`**——添加那 11 行提取代码。

---

## 7. 关键参考位置

| 内容 | 文件 | 行号 | 用途 |
|------|------|------|------|
| Qwen3 EAGLE3 提取（标准参考） | `src/models/qwen3.cpp` | 24-33 | 复制代码的模板 |
| Qwen3.5 EAGLE3 提取 | `src/models/qwen35.cpp` | 29-38 | 本次适配的实现 |
| EAGLE3 Encoder 实现 | `src/models/eagle3.cpp` | 3-38 | 理解 FC 融合层 |
| EAGLE3 Decoder 实现 | `src/models/eagle3.cpp` | 43-187 | 理解单层 Transformer |
| EAGLE3 检测逻辑 | `convert_hf_to_gguf.py` | ~2769 | draft 模型如何被识别 |
| EAGLE3 tensor 处理 | `convert_hf_to_gguf.py` | ~2920 | fc/d2t 等特殊 tensor 处理 |
| 已注册架构名列表 | `convert_hf_to_gguf.py` | ~2742-2754 | 确认 draft 架构名是否已注册 |

快速列出所有已支持 EAGLE3 的模型：

```bash
grep -rn "eagle3_extract" src/models/ | grep "eagle3_extract_0" | awk -F: '{print $1}' | sort -u
```

---

## 8. Draft 模型 Config 和 Tensor 速查

### Config.json 必填字段

```json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "num_hidden_layers": 1,
  "hidden_size": <与 target 相同>,
  "draft_vocab_size": <draft lm_head 词表大小>,
  "vocab_size": <target 词表大小>,
  "num_attention_heads": <draft 自身参数>,
  "num_key_value_heads": <draft 自身参数>,
  "head_dim": <draft 自身参数>,
  "intermediate_size": <draft 自身参数>
}
```

### Tensor 命名规则

| 训练输出名 | 转换后 GGUF 名 | 说明 |
|-----------|---------------|------|
| `fc.weight` | `fc.weight` | 直接保留 |
| `midlayer.input_layernorm.weight` | `blk.0.attn_norm.weight` | 自动重命名 |
| `midlayer.hidden_norm.weight` | `blk.0.hidden_norm.weight` | 特殊处理 |
| `midlayer.self_attn.q_proj.weight` | `blk.0.attn_q.weight` | 自动重命名 + permute |
| `midlayer.self_attn.k_proj.weight` | `blk.0.attn_k.weight` | 自动重命名 + permute |
| `midlayer.self_attn.v_proj.weight` | `blk.0.attn_v.weight` | 自动重命名 |
| `midlayer.self_attn.o_proj.weight` | `blk.0.attn_output.weight` | 自动重命名 |
| `midlayer.post_attention_layernorm.weight` | `blk.0.ffn_norm.weight` | 自动重命名 |
| `midlayer.mlp.*` | `blk.0.ffn_*` | 自动重命名 |
| `norm.weight` | `output_norm.weight` | 自动重命名 |
| `lm_head.weight` | `output.weight` | 自动重命名 |
| `d2t` | `d2t` | 保持 int64 |
| `t2d` | （跳过） | 不写入 GGUF |

### 提取层索引计算公式

```
extract_layers = [2, N // 2, N - 3]
```

| Target 模型 | N (层数) | extract_layers |
|------------|---------|----------------|
| Qwen3.5-4B | 32 | [2, 16, 29] |
| Qwen3-4B | 36 | [2, 18, 33] |
| Llama-3.1-8B | 32 | [2, 16, 29] |
| 假设 64 层模型 | 64 | [2, 32, 61] |
