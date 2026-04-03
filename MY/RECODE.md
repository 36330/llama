cd /home/easyai/llama/llama.cpp
rm -rf build  # 删除旧的构建目录以确保干净编译
mkdir build && cd build
【
cmake .. -G Ninja -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Debug  -DAMDGPU_TARGETS=gfx1103 -DGGML_CUDA_GRAPHS=OFF
cmake --build build --config Release

or

cmake .. -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Debug     
make -j 

gfx1151
# profiling 时
cmake .. -G Ninja -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA_GRAPHS=OFF

# 正式跑时
【rocm】
cmake .. -G Ninja -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_CUDA_GRAPHS=OFF -DGGML_HIP=ON -DGPU_TARGETS="gfx1151" -DLLAMA_CURL=OFF

cmake --build .

【vulkan】
cmake .. -G Ninja -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_CUDA_GRAPHS=OFF -DGGML_VULKAN=ON

cmake .. -G Ninja -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_CUDA_GRAPHS=OFF -DGGML_VULKAN=ON
】

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"
cmake -S . -B build

&& bash -c "cmake --build build --config Release -- -j 32"




// 投机解码
./build/bin/llama-cli -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf  -md /home/ljl/ljl_test/llama/models/Llama-3.2-1B-Instruct-Q8_0.gguf  -ngl 100 -ngld 10 --draft-max 8 

./build/bin/llama-bench -m /home/ljl/ljl_test/llama/models/Qwen3.5-27B-Q8_0.gguf -md /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf  -ngl 100 -ngld 100 --draft-max 8 



llama-cli -m /home/ljl/ljl_test/llama/models/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf -md /home/ljl/ljl_test/llama/models/Qwen3.5-0.8B-Q8_0.gguf  -n 100 -ngl 100 -ngld 100 --draft-max 5  --reasoning-budget 0

llama-cli -m /home/ljl/ljl_test/llama/models/Qwen3.5-27B-Q8_0.gguf -md /home/ljl/ljl_test/llama/models/Qwen3.5-0.8B-Q8_0.gguf  -n 100 -ngl 100 -ngld 100 --draft-max 5  --reasoning-budget 0

llama-bench -m /home/ljl/ljl_test/llama/models/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf -ngl 100 --reasoning-budget 0

/home/ljl/ljl_test/llama/build/bin/llama-speculative -m /home/ljl/ljl_test/llama/models/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf -md /home/ljl/ljl_test/llama/models/Qwen3.5-35B-A3B-Q8_0.gguf  -n 512 -ngl 100 -ngld 100 --draft-max 20 -p "解释勾股定理"

/home/ljl/ljl_test/llama/build/bin/llama-cli -m /home/ljl/ljl_test/llama/models/Qwen3.5-27B-Q8_0.gguf -md /home/ljl/ljl_test/llama/models/Qwen3.5-0.8B-Q8_0.gguf  -n 128 -ngl 100 -ngld 100 --draft-max 5 -p "解释勾股定理" --reasoning-budget 0
/home/ljl/ljl_test/llama/build/bin/llama-cli -m /home/ljl/ljl_test/llama/models/Qwen3.5-27B-Q8_0.gguf   -n 128 -ngl 100  -p "解释勾股定理" --reasoning-budget 0


//  -ngl 100 指定装载到gpu的层数
# Run inference directly in the terminal:
./build/bin/llama-cli -m /home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf -ngl 100 
# 

./build/bin/llama-bench -m /home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf -ngl 100 
#
./build/bin/llama-server -m /home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf -ngl 100 -c 16384 -ctk q8_0 -ctv q8_0 -fa



# 基本 profile
nsys profile -o llama_profile \
  ./build/bin/llama-cli -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf -ngl 100 -p "Hello" -n 100

ncu --graph-profiling node \
    --kernel-name "regex:mul_mat_vec_q" \
    --launch-skip 100 --launch-count 100 \
    --replay-mode application \
    -o llama_kernels \
    ./build/bin/llama-cli -m models/Llama-3.2-3B-Instruct-Q8_0.gguf -ngl 100 -p "hello"


ncu --graph-profiling node \
    --launch-skip 0 --launch-count 100 \
    -o llama_kernels \
    --replay-mode application \
    ./build/bin/llama-cli -m models/Llama-3.2-3B-Instruct-Q8_0.gguf -ngl 100 -p "hello"

rocprofv3 --kernel-trace --output-format csv -- llama-bench -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf -ngl 100 -p 512 -n 128 -fa 1

# 第一步：采集（会在当前目录生成 workloads/qwen_q8/ 目录）
rocprof-compute profile -n qwen_q8 -- llama-bench -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf -ngl 100 -p 512 -n 128

# 第二步：分析（指向采集生成的目录）
rocprof-compute analyze -p workloads/qwen_q8/

# 命令行分析
rocprof-compute analyze -p workloads/qwen_q8/

# 启动 Web UI 分析（类似 Nsight Compute 的 GUI）
rocprof-compute analyze -p workloads/qwen_q8/ --gui



// docker vllm

sudo docker stop vllm-jimoke
# 先退出容器
exit

// 重新进入容器
sudo docker start -ai vllm-jimoke
sudo docker exec -it vllm-jimoke /bin/bash

# 删掉旧容器，重建
sudo docker rm vllm-jimoke
sudo docker run -it \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --group-add sudo \
  -w /app/vllm/ \
  -v /home/ljl/ljl_test/llama/models:/models \
  --name vllm-jimoke \
  rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0 \
  /bin/bash

sudo docker cp /home/ljl/ljl_test/llama/MY/bench_vllm.sh vllm-jimoke:/models/bench_vllm.sh


vllm serve --model /models/Qwen3-4B --max-model-len 4096 --quantization awq


vllm bench latency --model /models/Qwen3-4B-Instruct-W8A8 --batch-size 1 --input-len 1024 --output-len 1024 --max-model-len 2048 --dtype float16 --enforce-eager --trust-remote-code

vllm bench latency --model /models/Qwen3.5-27B-GPTQ-Int4 --batch-size 1 --input-len 1024 --output-len 1024 --max-model-len 2048 --dtype float16 --enforce-eage --trust-remote-code

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-4B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'


//端口api
ss -tlnp | grep llama
kill $(lsof -t -i:8082)

121.89.86.47:6005
和
121.89.86.47:6006

# 模型 A - 端口 8081
./build/bin/llama-server \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -ngl 100 --host 0.0.0.0 --port 8081 --parallel 2 -c 16384 --repeat-penalty 1.1 --reasoning off --reasoning-format none --chat-template-kwargs '{"enable_thinking": false}' &

# 模型 B - 端口 8082
./build/bin/llama-server \
  -m models/Qwen3.5-27B-Q4_K_M.gguf \
  -ngl 100 --host 0.0.0.0 --port 8082 --parallel 2 -c 16384 --repeat-penalty 1.1 --reasoning off --reasoning-format none &

curl http://192.168.3.44:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b", "messages":[{"role":"user","content":"你好"}]}'

curl http://192.168.3.44:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-27b", "messages":[{"role":"user","content":"你好"}]}'



/root/autodl-tmp/llamacpp/llama.cpp/ggml/src/ggml-backend-reg.cpp

blas	通用 BLAS 线性代数库后端（OpenBLAS / MKL / Accelerate 等）
zendnn	AMD ZenDNN，针对 AMD CPU 优化的深度学习推理库
cann	华为昇腾 CANN（Compute Architecture for Neural Networks）
cuda	NVIDIA CUDA GPU 加速
hip	AMD ROCm/HIP GPU 加速（兼容 CUDA API 的 AMD 版本）
metal	Apple Metal GPU 加速（macOS/iOS）
rpc	远程过程调用后端，支持跨机器分布式推理
sycl	Intel oneAPI SYCL 后端（Intel GPU/CPU/FPGA）
vulkan	Vulkan GPU 计算（跨平台图形/计算 API）
virtgpu	VirtIO GPU 虚拟化后端
opencl	OpenCL 通用 GPU 计算（跨厂商）
hexagon	Qualcomm Hexagon DSP 加速（移动端）
musa	摩尔线程 MUSA GPU 加速（国产 GPU）
cpu	纯 CPU 后端（最基本的 fallback，始终可用）




## HIP 编译兼容性修复记录 (2026-03-13)

**文件**: `ggml/src/ggml-cuda/vendors/hip.h`
**原因**: 提交 `8af4dd8c` 添加 CUDA per-kernel timing 代码，缺少 HIP 宏映射导致 HIP 编译失败
**修改位置**: 第 58-68 行（event 相关宏定义区域）
**新增 5 个宏**:
```c
#define cudaEventCreate             hipEventCreate          // L58
#define cudaEventElapsedTime        hipEventElapsedTime     // L61
#define cudaStreamCaptureStatus     hipStreamCaptureStatus  // L66
#define cudaStreamIsCapturing       hipStreamIsCapturing    // L67
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone // L68
```

---

option(GGML_HIP_ROCWMMA_FATTN               "ggml: enable rocWMMA for FlashAttention"         OFF)
option(GGML_CUDA_FA                         "ggml: compile ggml FlashAttention CUDA kernels"  ON)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)







/root/autodl-tmp/llamacpp/llama.cpp/ggml/src/ggml-backend-reg.cpp

load_backend



/root/autodl-tmp/llamacpp/llama.cpp/src/llama-model.cpp

整个 offload 的核心在 llama-model.cpp 中的 load_tensors 函数。原理并不复杂，本质就是决定每个 tensor 放在哪个设备的内存里。





#### rocm后端设置

./llama-perplexity -m model.gguf -f wiki.test.raw



distrobox create --name llama-rocm-7.2 \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:rocm7-nightlies \
  --additional-flags "--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"


distrobox stop llama-rocm-7.2
distrobox rm llama-rocm-7.2


distrobox enter llama-rocm-7.2



cmake -B build-rocm7 \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.0/bin/hipcc \
  -DCMAKE_PREFIX_PATH=/opt/rocm-7.0

## 不同量化模型困惑度测试
llama-perplexity -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-BF16.gguf -f /home/ljl/ljl_test/llama/MY/wikitext-2-raw/wiki.test.raw



llama-cli --no-mmap -ngl 999 -fa 1 \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf \
  -p "Write a Strix Halo toolkit haiku."
llama-bench -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf -ngl 100 -fa 1 -ctk q8_0 -ctv q8_0

## 检测不到后端，宿主机执行 sudo chmod 666 /dev/kfd /dev/dri/*


ss -tlnp | grep llama
kill $(lsof -t -i:8082)

# 模型 A - 端口 8081
./build/bin/llama-server \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -ngl 100 --host 0.0.0.0 --port 8081 --parallel 2 -c 16384 --repeat-penalty 1.1 --reasoning off --reasoning-format none --chat-template-kwargs '{"enable_thinking": false}' &

# 模型 B - 端口 8082
./build/bin/llama-server \
  -m models/Qwen3.5-27B-Q4_K_M.gguf \
  -ngl 100 --host 0.0.0.0 --port 8082 --parallel 2 -c 16384 --repeat-penalty 1.1 --reasoning off --reasoning-format none &


# 模型 A - 端口 8081
./build/bin/llama-server \
  -m models/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  -ngl 100 --host 0.0.0.0 --port 8081 --parallel 2 -c 8192 --repeat-penalty 1.1 --reasoning off --reasoning-format none --chat-template-kwargs '{"enable_thinking": true}' &

# 模型 B - 端口 8082
./build/bin/llama-server \
  -m models/Qwen3.5-0.8B-Q8_0.gguf \
  -ngl 10 --host 0.0.0.0 --port 8082 --parallel 2 -c 64 --repeat-penalty 1.1 --reasoning off --reasoning-format none &


  ./build/bin/llama-server \
  -m models/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  -ngl 100 --host 0.0.0.0 --port 8081 --parallel 2 -c 8192 --repeat-penalty 1.1 --reasoning on --chat-template-kwargs '{"enable_thinking": true}' &


       


## 转换
python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/modelss/qwen3-30b-eagle" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/Qwen3-30b-a3b" \
    --outfile "/home/ljl/ljl_test/llama/models"
    
python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/modelss/llama3.1-8b-eagle" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/llama-3.1-8B" \
    --outfile "/home/ljl/ljl_test/llama/models"

python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/llama/models/qwen3-32b-eagle3" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/qwen3-32b-tf" \
    --outfile "/home/ljl/ljl_test/llama/models"

python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/llama/models/qwen3-30b-a3b-eagle" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/Qwen3-30B-A3B-Thinking-2507-tf" \
    --outfile "/home/ljl/ljl_test/llama/models"


python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/llama/models/Qwen3-8B_eagle3" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/Qwen3-8B-tf" \
    --outfile "/home/ljl/ljl_test/llama/models"

python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/llama/models/qwen3-30b-a3b-eagle" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/Qwen3-30B-A3B-Thinking-2507-tf" \
    --outfile "/home/ljl/ljl_test/llama/models"


python convert_hf_to_gguf.py \
    "/home/ljl/ljl_test/llama/models/qwen3-30b-a3b-eagle" \
    --outtype f16 \
    --target-model-dir "/home/ljl/ljl_test/llama/models/Qwen3-30B-A3B-Thinking-2507-tf" \
    --outfile "/home/ljl/ljl_test/llama/models"






## 量化
./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/gpt-oss-795M-120b-eagle-F16.gguf \
  /home/ljl/ljl_test/llama/models/gpt-oss-795M-120b-eagle-Q4_K_M.gguf \
  Q4_K_M

./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/eagle3-qwen3.5-9b-eagle.gguf \
  /home/ljl/ljl_test/llama/models/eagle3-qwen3.5-9b-eagle_Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/qwen3-728M-32b-eagle3-F16.gguf \
  /home/ljl/ljl_test/llama/models/qwen3-728M-32b-eagle3-Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/qwen3-a3B-30b-eagle-F16.gguf \
  /home/ljl/ljl_test/llama/models/qwen3-a3B-30b-eagle-Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/Qwen3-400M-8B_eagle3-F16.gguf \
  /home/ljl/ljl_test/llama/models/Qwen3-400M-8B_eagle3--Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/Qwen3-400M-8B_eagle3-F16.gguf \
  /home/ljl/ljl_test/llama/models/Qwen3-400M-8B_eagle3--Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /home/ljl/ljl_test/llama/models/qwen3-a3B-30b-eagle-F16.gguf \
  /home/ljl/ljl_test/llama/models/qwen3-a3B-30b-eagle-Q8_0.gguf \
  Q8_0



## 测试
./build/bin/llama-bench \
  -m /home/ljl/ljl_test/llama/models/gpt-oss-120b-gguf-q4ks-AutoRound/gpt-oss-120b-128x3.0B-Q4_K_S.gguf \
  -ngl 99 -pg 1,1 -pg 2,1 -pg 4,1 -pg 8,1


# 无投机解码 baseline
./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/gpt-oss-120b-gguf-q4ks-AutoRound/gpt-oss-120b-128x3.0B-Q4_K_S.gguf \
  -p "Write a quicksort algorithm in Python" \
  -n 256 --temp 0 --seed 42 -ngl 99

# 有 EAGLE3
./build/bin/llama-speculative-simple \
  -m /home/ljl/ljl_test/llama/models/gpt-oss-120b-gguf-q4ks-AutoRound/gpt-oss-120b-128x3.0B-Q4_K_S.gguf \
  -md /home/ljl/ljl_test/llama/models/gpt-oss-795M-120b-eagle-Q4_K_M.gguf \
  --eagle3 -p "Write a quicksort algorithm in Python" \
  -n 256 --draft 4 --temp 0 --seed 42 -ngl 99 -ngld 99

for prompt in \
    "Write a quicksort algorithm in Python. Write code only." \
    "Explain the Pythagorean theorem" \
    "Plan a 1 day trip to DC"; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/gpt-oss-120b-gguf-q4ks-AutoRound/gpt-oss-120b-128x3.0B-Q4_K_S.gguf \
      -md /home/ljl/ljl_test/llama/models/llama3.1-425M-8b-eagle_Q4_K_M.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done





# 无投机解码 baseline
./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf \
  -p "Write a quicksort algorithm in Python" \
  -n 256 --temp 0 --seed 42 -ngl 99

# 有 EAGLE3
./build/bin/llama-speculative-simple \
  -m /home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf \
  -md /home/ljl/ljl_test/llama/models/qwen3-4b-eagle_f16.gguf \
  --eagle3 -p "Write a quicksort algorithm in Python" \
  -n 256 --draft 8 --temp 0 --seed 42 -ngl 99 -ngld 99

for prompt in \
    "Write a quicksort algorithm in Python" ; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf \
      -md /home/ljl/ljl_test/llama/models/qwen3-4b-eagle_f16.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done


# 无投机解码 baseline
./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf \
  -p "Write a quicksort algorithm in Python" \
  -n 256 --temp 0 --seed 42 -ngl 99

# 有 EAGLE3
./build/bin/llama-speculative-simple \
  -m /home/ljl/ljl_test/llama/models/gpt-oss-120b-gguf-q4ks-AutoRound/gpt-oss-120b-128x3.0B-Q4_K_S.gguf \
  -md /home/ljl/ljl_test/llama/models/llama3.1-425M-8b-eagle_Q4_K_M.gguf \
  --eagle3 -p "Write a quicksort algorithm in Python" \
  -n 256 --draft 4 --temp 0 --seed 42 -ngl 99 -ngld 99

for prompt in \
    "Write a quicksort algorithm in Python."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf \
      -md /home/ljl/ljl_test/llama/models/qwen3-30b-eagle_Q4_K_M.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 8 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done


./build/bin/llama-bench \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-27B-Q8_0.gguf \
  -ngl 99 -p 1 -p 2 -p 4 -p 8

现在我需要正式实现qwen3.5的eagle3推理了，这是之前的plan /home/ljl/ljl_test/llama/MY/eagle3_qwen35_plan.md   然后/home/ljl/ljl_test/modelss/epoch_2_step_12000  这是我训练好的qwen3.5-4b的eagle模型  就用这个和/home/ljl/ljl_test/llama/models/Qwen3-4B-Q8_0.gguf  分别作为draft和target,但是现在的draft还是需要转成gguf才行，可能也需要更改




################################
for prompt in \
    "Write a quicksort algorithm in Python"; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-BF16.gguf \
      -md /home/ljl/ljl_test/llama/models/eagle3-qwen35-4b-draft-f16.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-BF16.gguf  \
  -p "Write a quicksort algorithm in Python" \
  -n 256 --temp 0 --seed 42 -ngl 99


################################
for prompt in \
    "Write a quicksort algorithm in Python"; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf \
      -md /home/ljl/ljl_test/llama/models/eagle3-qwen35-4b-draft-Q8_0.gguf \
      --eagle3 -p "$prompt" -n 512 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q8_0.gguf  \
  -p "Write a quicksort algorithm in Python" \
  -n 512 --temp 0 --seed 42 -ngl 16


################################
for prompt in \
    "Write a function to remove all elements from a given list present in another list."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf \
      -md /home/ljl/ljl_test/llama/models/eagle3-qwen35-4b-draft-Q4_K_M.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done



./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf  \
  -p "Write a function to remove all elements from a given list present in another list." \
  -n 256 --temp 0 --seed 42 -ngl 99








################################
for prompt in \
    "Write a function to remove all elements from a given list present in another list."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3.5-9B-BF16.gguf \
      -md /home/ljl/ljl_test/llama/models/eagle3-qwen3.5-9b-eagle.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-9B-BF16.gguf  \
  -p "Write a function to remove all elements from a given list present in another list." \
  -n 256 --temp 0 --seed 42 -ngl 99



################################
for prompt in \
    "Write a quicksort algorithm in Python."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf \
      -md /home/ljl/ljl_test/llama/models/qwen3-30b-eagle_Q4_K_M.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 1 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf  \
  -p "Write a function to remove all elements from a given list present in another list." \
  -n 256 --temp 0 --seed 42 -ngl 99



################################
for prompt in \
    "Write a function to remove all elements from a given list present in another list and use three method."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3-32B-Q8_0.gguf \
      -md /home/ljl/ljl_test/llama/models/qwen3-728M-32b-eagle3-Q8_0.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3-32B-Q8_0.gguf  \
  -p "Write a function to remove all elements from a given list present in another list and use three method." \
  -n 256 --temp 0 --seed 42 -ngl 99



################################
for prompt in \
    "Write a function to remove all elements from a given list present in another list and use three method."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf \
      -md /home/ljl/ljl_test/llama/models/qwen3-a3B-30b-eagle-Q8_0.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf  \
  -p "Write a function to remove all elements from a given list present in another list and use three method." \
  -n 256 --temp 0 --seed 42 -ngl 99



################################
for prompt in \
    "Write a quicksort algorithm in Python. Write code only."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Llama-3.1-8B-Instruct-f16.gguf \
      -md /home/ljl/ljl_test/llama/models/llama3.1-425M-8b-eagle-F16.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 4 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Llama-3.1-8B-Instruct-f16.gguf  \
  -p "Write a function to remove all elements from a given list present in another list and use three method." \
  -n 256 --temp 0 --seed 42 -ngl 99



################################
for prompt in \
    "Write a function to remove all elements from a given list present in another list and use three method."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3-8B-Q8_0.gguf \
      -md /home/ljl/ljl_test/llama/models/Qwen3-400M-8B_eagle3--Q8_0.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3-8B-Q8_0.gguf  \
  -p "Write a function to remove all elements from a given list present in another list and use three method." \
  -n 256 --temp 0 --seed 42 -ngl 99










################################
for prompt in \
    "Write a function to remove all elements from a given list present in another list."; do
  echo "=== Prompt: $prompt ==="
    ./build/bin/llama-speculative-simple \
      -m /data/data/com.termux/files/home/ljl/llama-master/models/Qwen3.5-4B-Q8_0.gguf \
      -md /data/data/com.termux/files/home/ljl/llama-master/models/eagle3-qwen35-4b-draft-Q8_0.gguf \
      --eagle3 -p "$prompt" -n 256 --draft 4 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done





./build/bin/llama-cli \
  -m /data/data/com.termux/files/home/ljl/llama-master/models/Qwen3.5-4B-Q8_0.gguf  \
  -p "Write a function to remove all elements from a given list present in another list." \
  -n 256 --temp 0 --seed 42 



## 编译

cmake -S /home/ljl/ljl_test/llama \
      -B /home/ljl/ljl_test/llama/build \
      -G Ninja \
      -DLLAMA_OPENSSL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMDGPU_TARGETS=gfx1151 \
      -DGGML_HIP_ROCWMMA_FATTN=OFF \
      -DGGML_CUDA_GRAPHS=OFF \
      -DGGML_VULKAN=ON


cmake --build /home/ljl/ljl_test/llama/build --target llama-speculative-simple -j




clear

for prompt in \
    "Write a function to remove all elements from a given list present in another list and use three method."; do
  echo "=== Prompt: $prompt ==="
    /home/ljl/ljl_test/llama/build/bin/llama-speculative-simple \
      -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf \
      -md /home/ljl/ljl_test/llama/models/eagle3-qwen35-4b-draft-Q4_K_M.gguf \
      --eagle3 -p "$prompt" -n 1024 --draft 2 \
      --temp 0 --top-k 1 --seed 42 -ngl 99 -ngld 99 
done

./build/bin/llama-cli \
  -m /home/ljl/ljl_test/llama/models/Qwen3.5-4B-Q4_K_M.gguf  \
  -p "Write a function to remove all elements from a given list present in another list and use three method." \
  -n 1024 --temp 0 --seed 42 -ngl 99