# NER Evaluation Metrics: Qwen vs Otter

## English (ENG)

| Model | F1 Score | Latency (ms/example) | VRAM (GB) |
|-------|----------|----------------------|-----------|
| Qwen3-0.6B | 0.197 | 1029.88 | 1.25 |
| Qwen3-4B | 0.475 | 1111.13 | 7.73 |
| Otter | 0.7687 | 18.49 | 1.43 |

## German (DEU)

| Model | F1 Score | Latency (ms/example) | VRAM (GB) |
|-------|----------|----------------------|-----------|
| Qwen3-0.6B | 0.241 | 939.63 | 1.20 |
| Qwen3-4B | 0.484 | 1316.19 | 7.75 |
| Otter | 0.7088 | 18.26 | 1.44 |

## Russian (RUS)

| Model | F1 Score | Latency (ms/example) | VRAM (GB) |
|-------|----------|----------------------|-----------|
| Qwen3-0.6B | 0.043 | 1526.13 | 1.25 |
| Qwen3-4B | 0.274 | 1338.06 | 7.75 |
| Otter | 0.5884 | 18.32 | 1.45 |

---

## Lexical Overlap (Jensen-Shannon Divergence)

| Model | ENG | DEU | RUS |
|-------|-----|-----|-----|
| FiNERweb | 0.1528 ± 0.0007 | 0.1697 ± 0.0008 | 0.2351 ± 0.0007 |
| Euro-GLiNER-x | 0.1568 ± 0.0015 | 0.1360 ± 0.0018 | - |
| PileNER | 0.1603 ± 0.0011 | - | - |

## Embedding Overlap (F-distance)

| Model | ENG | DEU | RUS |
|-------|-----|-----|-----|
| FiNERweb | 2.339 ± 0.022 | 1.844 ± 0.031 | 2.413 ± 0.022 |
| Euro-GLiNER-x | 0.959 ± 0.018 | 0.379 ± 0.006 | - |
| PileNER | 3.545 ± 0.028 | - | - |
