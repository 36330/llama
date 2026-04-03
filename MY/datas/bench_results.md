# Qwen 模型 Benchmark 结果

测试时间: 2026-03-14 10:58:28
每个模型测试 3 次

| model | size | params | backend | ngl | test | t/s | run |
| ----- | ---: | -----: | ------- | --: | ---: | --: | --: |
| qwen3 32B Q2_K - Medium        |  11.49 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        247.87 ± 0.00  1/3 |
| qwen3 32B Q2_K - Medium        |  11.49 GiB |    32.76 B | Vulkan     | 100 |           tg128 |         17.46 ± 0.00  1/3 |
| qwen3 32B Q2_K - Medium        |  11.49 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        247.11 ± 0.00  2/3 |
| qwen3 32B Q2_K - Medium        |  11.49 GiB |    32.76 B | Vulkan     | 100 |           tg128 |         17.42 ± 0.00  2/3 |
| qwen3 32B Q2_K - Medium        |  11.49 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        246.64 ± 0.00  3/3 |
| qwen3 32B Q2_K - Medium        |  11.49 GiB |    32.76 B | Vulkan     | 100 |           tg128 |         17.40 ± 0.00  3/3 |
| qwen3 32B Q4_K - Medium        |  18.40 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        242.75 ± 0.00  1/3 |
| qwen3 32B Q4_K - Medium        |  18.40 GiB |    32.76 B | Vulkan     | 100 |           tg128 |         10.76 ± 0.00  1/3 |
| qwen3 32B Q4_K - Medium        |  18.40 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        248.83 ± 0.00  2/3 |
| qwen3 32B Q4_K - Medium        |  18.40 GiB |    32.76 B | Vulkan     | 100 |           tg128 |         11.01 ± 0.00  2/3 |
| qwen3 32B Q4_K - Medium        |  18.40 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        248.91 ± 0.00  3/3 |
| qwen3 32B Q4_K - Medium        |  18.40 GiB |    32.76 B | Vulkan     | 100 |           tg128 |         11.02 ± 0.00  3/3 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        233.63 ± 0.00  1/3 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | Vulkan     | 100 |           tg128 |          7.99 ± 0.00  1/3 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        242.61 ± 0.00  2/3 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | Vulkan     | 100 |           tg128 |          8.25 ± 0.00  2/3 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        242.85 ± 0.00  3/3 |
| qwen3 32B Q6_K                 |  25.03 GiB |    32.76 B | Vulkan     | 100 |           tg128 |          8.25 ± 0.00  3/3 |
| qwen3 32B Q8_0                 |  32.42 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        251.53 ± 0.00  1/3 |
| qwen3 32B Q8_0                 |  32.42 GiB |    32.76 B | Vulkan     | 100 |           tg128 |          6.41 ± 0.00  1/3 |
| qwen3 32B Q8_0                 |  32.42 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        250.40 ± 0.00  2/3 |
| qwen3 32B Q8_0                 |  32.42 GiB |    32.76 B | Vulkan     | 100 |           tg128 |          6.39 ± 0.00  2/3 |
| qwen3 32B Q8_0                 |  32.42 GiB |    32.76 B | Vulkan     | 100 |           pp512 |        250.12 ± 0.00  3/3 |
| qwen3 32B Q8_0                 |  32.42 GiB |    32.76 B | Vulkan     | 100 |           tg128 |          6.38 ± 0.00  3/3 |
| qwen3 4B Q2_K - Medium         |   1.55 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1806.23 ± 0.00  1/3 |
| qwen3 4B Q2_K - Medium         |   1.55 GiB |     4.02 B | Vulkan     | 100 |           tg128 |        102.45 ± 0.00  1/3 |
| qwen3 4B Q2_K - Medium         |   1.55 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1807.29 ± 0.00  2/3 |
| qwen3 4B Q2_K - Medium         |   1.55 GiB |     4.02 B | Vulkan     | 100 |           tg128 |        103.00 ± 0.00  2/3 |
| qwen3 4B Q2_K - Medium         |   1.55 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1802.53 ± 0.00  3/3 |
| qwen3 4B Q2_K - Medium         |   1.55 GiB |     4.02 B | Vulkan     | 100 |           tg128 |        102.67 ± 0.00  3/3 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1797.99 ± 0.00  1/3 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         77.39 ± 0.00  1/3 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1815.48 ± 0.00  2/3 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         77.31 ± 0.00  2/3 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1811.54 ± 0.00  3/3 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         77.26 ± 0.00  3/3 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1764.02 ± 0.00  1/3 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         60.58 ± 0.00  1/3 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1757.83 ± 0.00  2/3 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         60.54 ± 0.00  2/3 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1758.39 ± 0.00  3/3 |
| qwen3 4B Q6_K                  |   3.07 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         60.67 ± 0.00  3/3 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1897.30 ± 0.00  1/3 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         48.48 ± 0.00  1/3 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1897.22 ± 0.00  2/3 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         48.65 ± 0.00  2/3 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | Vulkan     | 100 |           pp512 |       1911.51 ± 0.00  3/3 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | Vulkan     | 100 |           tg128 |         48.62 ± 0.00  3/3 |
| qwen3 8B Q2_K - Medium         |   3.05 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1005.62 ± 0.00  1/3 |
| qwen3 8B Q2_K - Medium         |   3.05 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         62.99 ± 0.00  1/3 |
| qwen3 8B Q2_K - Medium         |   3.05 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1007.61 ± 0.00  2/3 |
| qwen3 8B Q2_K - Medium         |   3.05 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         63.16 ± 0.00  2/3 |
| qwen3 8B Q2_K - Medium         |   3.05 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1001.89 ± 0.00  3/3 |
| qwen3 8B Q2_K - Medium         |   3.05 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         63.06 ± 0.00  3/3 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1018.98 ± 0.00  1/3 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         43.71 ± 0.00  1/3 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1016.50 ± 0.00  2/3 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         43.65 ± 0.00  2/3 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1015.74 ± 0.00  3/3 |
| qwen3 8B Q4_K - Medium         |   4.68 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         43.70 ± 0.00  3/3 |
| qwen3 8B Q6_K                  |   6.26 GiB |     8.19 B | Vulkan     | 100 |           pp512 |        975.22 ± 0.00  1/3 |
| qwen3 8B Q6_K                  |   6.26 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         34.12 ± 0.00  1/3 |
| qwen3 8B Q6_K                  |   6.26 GiB |     8.19 B | Vulkan     | 100 |           pp512 |        977.71 ± 0.00  2/3 |
| qwen3 8B Q6_K                  |   6.26 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         34.10 ± 0.00  2/3 |
| qwen3 8B Q6_K                  |   6.26 GiB |     8.19 B | Vulkan     | 100 |           pp512 |        978.27 ± 0.00  3/3 |
| qwen3 8B Q6_K                  |   6.26 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         34.14 ± 0.00  3/3 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1038.42 ± 0.00  1/3 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         26.77 ± 0.00  1/3 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1034.56 ± 0.00  2/3 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         26.86 ± 0.00  2/3 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | Vulkan     | 100 |           pp512 |       1031.46 ± 0.00  3/3 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | Vulkan     | 100 |           tg128 |         26.84 ± 0.00  3/3 |
| qwen3moe 235B.A22B Q2_K - Small |  74.33 GiB |   235.09 B | Vulkan     | 100 |           pp512 |        156.34 ± 0.00  1/3 |
| qwen3moe 235B.A22B Q2_K - Small |  74.33 GiB |   235.09 B | Vulkan     | 100 |           tg128 |         19.14 ± 0.00  1/3 |
| qwen3moe 235B.A22B Q2_K - Small |  74.33 GiB |   235.09 B | Vulkan     | 100 |           pp512 |        156.24 ± 0.00  2/3 |
| qwen3moe 235B.A22B Q2_K - Small |  74.33 GiB |   235.09 B | Vulkan     | 100 |           tg128 |         19.10 ± 0.00  2/3 |
| qwen3moe 235B.A22B Q2_K - Small |  74.33 GiB |   235.09 B | Vulkan     | 100 |           pp512 |        155.78 ± 0.00  3/3 |
| qwen3moe 235B.A22B Q2_K - Small |  74.33 GiB |   235.09 B | Vulkan     | 100 |           tg128 |         19.14 ± 0.00  3/3 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB |   235.09 B | Vulkan     |  30 |           pp512 |         14.47 ± 0.00  1/3 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB |   235.09 B | Vulkan     |  30 |           tg128 |          2.76 ± 0.00  1/3 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB |   235.09 B | Vulkan     |  30 |           pp512 |         14.35 ± 0.00  2/3 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB |   235.09 B | Vulkan     |  30 |           tg128 |          2.86 ± 0.00  2/3 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB |   235.09 B | Vulkan     |  30 |           pp512 |         13.85 ± 0.00  3/3 |
| qwen3moe 235B.A22B Q4_K - Medium | 123.49 GiB |   235.09 B | Vulkan     |  30 |           tg128 |          2.81 ± 0.00  3/3 |
