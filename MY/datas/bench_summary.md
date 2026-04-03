# Qwen 模型 Benchmark 汇总 (3次平均)

| model | size | params | backend | ngl | test | avg t/s |
| ----- | ---: | -----: | ------- | --: | ---: | ------: |
| qwen3 32B Q2_K - Medium | 11.49 GiB | 32.76 B | Vulkan | 100 | pp512 | 247.21 |
| qwen3 32B Q2_K - Medium | 11.49 GiB | 32.76 B | Vulkan | 100 | tg128 | 17.43 |
| qwen3 32B Q4_K - Medium | 18.40 GiB | 32.76 B | Vulkan | 100 | pp512 | 246.83 |
| qwen3 32B Q4_K - Medium | 18.40 GiB | 32.76 B | Vulkan | 100 | tg128 | 10.93 |
| qwen3 32B Q6_K | 25.03 GiB | 32.76 B | Vulkan | 100 | pp512 | 239.70 |
| qwen3 32B Q6_K | 25.03 GiB | 32.76 B | Vulkan | 100 | tg128 | 8.16 |
| qwen3 32B Q8_0 | 32.42 GiB | 32.76 B | Vulkan | 100 | pp512 | 250.68 |
| qwen3 32B Q8_0 | 32.42 GiB | 32.76 B | Vulkan | 100 | tg128 | 6.39 |
| qwen3 4B Q2_K - Medium | 1.55 GiB | 4.02 B | Vulkan | 100 | pp512 | 1805.35 |
| qwen3 4B Q2_K - Medium | 1.55 GiB | 4.02 B | Vulkan | 100 | tg128 | 102.71 |
| qwen3 4B Q4_K - Medium | 2.32 GiB | 4.02 B | Vulkan | 100 | pp512 | 1808.34 |
| qwen3 4B Q4_K - Medium | 2.32 GiB | 4.02 B | Vulkan | 100 | tg128 | 77.32 |
| qwen3 4B Q6_K | 3.07 GiB | 4.02 B | Vulkan | 100 | pp512 | 1760.08 |
| qwen3 4B Q6_K | 3.07 GiB | 4.02 B | Vulkan | 100 | tg128 | 60.60 |
| qwen3 4B Q8_0 | 3.98 GiB | 4.02 B | Vulkan | 100 | pp512 | 1902.01 |
| qwen3 4B Q8_0 | 3.98 GiB | 4.02 B | Vulkan | 100 | tg128 | 48.58 |
| qwen3 8B Q2_K - Medium | 3.05 GiB | 8.19 B | Vulkan | 100 | pp512 | 1005.04 |
| qwen3 8B Q2_K - Medium | 3.05 GiB | 8.19 B | Vulkan | 100 | tg128 | 63.07 |
| qwen3 8B Q4_K - Medium | 4.68 GiB | 8.19 B | Vulkan | 100 | pp512 | 1017.07 |
| qwen3 8B Q4_K - Medium | 4.68 GiB | 8.19 B | Vulkan | 100 | tg128 | 43.69 |
| qwen3 8B Q6_K | 6.26 GiB | 8.19 B | Vulkan | 100 | pp512 | 977.07 |
| qwen3 8B Q6_K | 6.26 GiB | 8.19 B | Vulkan | 100 | tg128 | 34.12 |
| qwen3 8B Q8_0 | 8.11 GiB | 8.19 B | Vulkan | 100 | pp512 | 1034.81 |
| qwen3 8B Q8_0 | 8.11 GiB | 8.19 B | Vulkan | 100 | tg128 | 26.82 |
| qwen3moe 235B.A22B Q2_K - Small | 74.33 GiB | 235.09 B | Vulkan | 100 | pp512 | 156.12 |
| qwen3moe 235B.A22B Q2_K - Small | 74.33 GiB | 235.09 B | Vulkan | 100 | tg128 | 19.13 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB | 235.09 B | Vulkan | 30 | pp512 | 14.22 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB | 235.09 B | Vulkan | 30 | tg128 | 2.81 |
