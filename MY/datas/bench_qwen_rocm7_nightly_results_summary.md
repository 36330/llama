# Qwen 模型 Benchmark 汇总 (3次平均)

| model | size | params | backend | ngl | test | stage | avg t/s |
| ----- | ---: | -----: | ------- | --: | ---: | ------: | ------: |
| qwen3 32B Q2_K - Medium | 11.49 GiB | 32.76 B | ROCm       | 100 | 1 | pp512 | 322.71 |
| qwen3 32B Q2_K - Medium | 11.49 GiB | 32.76 B | ROCm       | 100 | 1 | tg128 | 16.50 |
| qwen3 32B Q4_K - Medium | 18.40 GiB | 32.76 B | ROCm       | 100 | 1 | pp512 | 330.10 |
| qwen3 32B Q4_K - Medium | 18.40 GiB | 32.76 B | ROCm       | 100 | 1 | tg128 | 10.92 |
| qwen3 32B Q6_K | 25.03 GiB | 32.76 B | ROCm       | 100 | 1 | pp512 | 293.85 |
| qwen3 32B Q6_K | 25.03 GiB | 32.76 B | ROCm       | 100 | 1 | tg128 | 8.11 |
| qwen3 4B Q2_K - Medium | 1.55 GiB | 4.02 B | ROCm       | 100 | 1 | pp512 | 2365.79 |
| qwen3 4B Q2_K - Medium | 1.55 GiB | 4.02 B | ROCm       | 100 | 1 | tg128 | 92.11 |
| qwen3 4B Q4_K - Medium | 2.32 GiB | 4.02 B | ROCm       | 100 | 1 | pp512 | 2359.73 |
| qwen3 4B Q4_K - Medium | 2.32 GiB | 4.02 B | ROCm       | 100 | 1 | tg128 | 67.74 |
| qwen3 4B Q6_K | 3.07 GiB | 4.02 B | ROCm       | 100 | 1 | pp512 | 2231.25 |
| qwen3 4B Q6_K | 3.07 GiB | 4.02 B | ROCm       | 100 | 1 | tg128 | 56.38 |
| qwen3 4B Q8_0 | 3.98 GiB | 4.02 B | ROCm       | 100 | 1 | pp512 | 2333.64 |
| qwen3 4B Q8_0 | 3.98 GiB | 4.02 B | ROCm       | 100 | 1 | tg128 | 45.76 |
| qwen3 8B Q2_K - Medium | 3.05 GiB | 8.19 B | ROCm       | 100 | 1 | pp512 | 1020.96 |
| qwen3 8B Q2_K - Medium | 3.05 GiB | 8.19 B | ROCm       | 100 | 1 | tg128 | 58.34 |
| qwen3 8B Q4_K - Medium | 4.68 GiB | 8.19 B | ROCm       | 100 | 1 | pp512 | 1218.59 |
| qwen3 8B Q4_K - Medium | 4.68 GiB | 8.19 B | ROCm       | 100 | 1 | tg128 | 41.36 |
| qwen3 8B Q6_K | 6.26 GiB | 8.19 B | ROCm       | 100 | 1 | pp512 | 858.39 |
| qwen3 8B Q6_K | 6.26 GiB | 8.19 B | ROCm       | 100 | 1 | tg128 | 32.00 |
| qwen3 8B Q8_0 | 8.11 GiB | 8.19 B | ROCm       | 100 | 1 | pp512 | 1370.47 |
| qwen3 8B Q8_0 | 8.11 GiB | 8.19 B | ROCm       | 100 | 1 | tg128 | 26.12 |
| qwen35 27B Q2_K - Medium | 10.43 GiB | 26.90 B | ROCm       | 100 | 1 | pp512 | 372.13 |
| qwen35 27B Q2_K - Medium | 10.43 GiB | 26.90 B | ROCm       | 100 | 1 | tg128 | 16.53 |
| qwen35 27B Q4_K - Medium | 15.58 GiB | 26.90 B | ROCm       | 100 | 1 | pp512 | 370.09 |
| qwen35 27B Q4_K - Medium | 15.58 GiB | 26.90 B | ROCm       | 100 | 1 | tg128 | 11.80 |
| qwen35 27B Q6_K | 20.90 GiB | 26.90 B | ROCm       | 100 | 1 | pp512 | 346.69 |
| qwen35 27B Q6_K | 20.90 GiB | 26.90 B | ROCm       | 100 | 1 | tg128 | 9.21 |
| qwen35 27B Q8_0 | 26.62 GiB | 26.90 B | ROCm       | 100 | 1 | pp512 | 368.69 |
| qwen35 27B Q8_0 | 26.62 GiB | 26.90 B | ROCm       | 100 | 1 | tg128 | 7.38 |
| qwen35 4B Q2_K - Medium | 1.80 GiB | 4.21 B | ROCm       | 100 | 1 | pp512 | 2042.07 |
| qwen35 4B Q2_K - Medium | 1.80 GiB | 4.21 B | ROCm       | 100 | 1 | tg128 | 70.57 |
| qwen35 4B Q4_K - Medium | 2.54 GiB | 4.21 B | ROCm       | 100 | 1 | pp512 | 1929.04 |
| qwen35 4B Q4_K - Medium | 2.54 GiB | 4.21 B | ROCm       | 100 | 1 | tg128 | 57.74 |
| qwen35 4B Q6_K | 3.27 GiB | 4.21 B | ROCm       | 100 | 1 | pp512 | 1835.96 |
| qwen35 4B Q6_K | 3.27 GiB | 4.21 B | ROCm       | 100 | 1 | tg128 | 48.46 |
| qwen35 4B Q8_0 | 4.16 GiB | 4.21 B | ROCm       | 100 | 1 | pp512 | 2070.25 |
| qwen35 4B Q8_0 | 4.16 GiB | 4.21 B | ROCm       | 100 | 1 | tg128 | 40.34 |
| qwen35 9B Q2_K - Medium | 3.83 GiB | 8.95 B | ROCm       | 100 | 1 | pp512 | 1006.99 |
| qwen35 9B Q2_K - Medium | 3.83 GiB | 8.95 B | ROCm       | 100 | 1 | tg128 | 46.53 |
| qwen35 9B Q4_K - Medium | 5.28 GiB | 8.95 B | ROCm       | 100 | 1 | pp512 | 1061.64 |
| qwen35 9B Q4_K - Medium | 5.28 GiB | 8.95 B | ROCm       | 100 | 1 | tg128 | 35.58 |
| qwen35 9B Q6_K | 6.94 GiB | 8.95 B | ROCm       | 100 | 1 | pp512 | 844.73 |
| qwen35 9B Q6_K | 6.94 GiB | 8.95 B | ROCm       | 100 | 1 | tg128 | 28.61 |
| qwen35 9B Q8_0 | 8.86 GiB | 8.95 B | ROCm       | 100 | 1 | pp512 | 1243.70 |
| qwen35 9B Q8_0 | 8.86 GiB | 8.95 B | ROCm       | 100 | 1 | tg128 | 23.68 |
| qwen3moe 30B.A3B Q2_K - Medium | 10.48 GiB | 30.53 B | ROCm       | 100 | 1 | pp512 | 1091.10 |
| qwen3moe 30B.A3B Q2_K - Medium | 10.48 GiB | 30.53 B | ROCm       | 100 | 1 | tg128 | 92.72 |
| qwen3moe 30B.A3B Q4_K - Medium | 17.28 GiB | 30.53 B | ROCm       | 100 | 1 | pp512 | 1203.86 |
| qwen3moe 30B.A3B Q4_K - Medium | 17.28 GiB | 30.53 B | ROCm       | 100 | 1 | tg128 | 73.23 |
| qwen3moe 30B.A3B Q6_K | 23.36 GiB | 30.53 B | ROCm       | 100 | 1 | pp512 | 941.41 |
| qwen3moe 30B.A3B Q6_K | 23.36 GiB | 30.53 B | ROCm       | 100 | 1 | tg128 | 61.78 |
