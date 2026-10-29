# VectorForge HNSW Benchmark Results

## Test environment

- Vector dimension: 64
- Metric: L2
- HNSW M: 16
- Default ef_construction: 100
- Seed: 20261029
- Query count: 200 for the 5K, 10K, 25K, 50K and 100K runs
- Note: Results use synthetic Gaussian vectors, not actual CLIP image embeddings. Timings are machine-dependent.

## Dataset scaling

| Dataset size | Index build (ms) | Exact p50 (ms) | Exact p95 (ms) |
|---:|---:|---:|---:|
| 5,000 | 748.401 | 0.121 | 0.181 |
| 10,000 | 975.558 | 0.127 | 0.154 |
| 25,000 | 3,163.242 | 0.315 | 0.474 |
| 50,000 | 8,092.135 | 0.739 | 1.085 |
| 100,000 | 20,355.777 | 1.715 | 2.176 |

## HNSW search results

Recall@10 and p50 latency for the default construction configuration (`ef_construction=100`):

| Dataset | ef | Recall@10 | p50 (ms) | p95 (ms) | Speedup vs exact |
|---:|---:|---:|---:|---:|---:|
| 5K | 16 | 0.661 | 0.028 | 0.074 | 4.325x |
| 5K | 32 | 0.814 | 0.050 | 0.087 | 2.396x |
| 5K | 64 | 0.924 | 0.093 | 0.142 | 1.297x |
| 5K | 128 | 0.981 | 0.186 | 0.254 | 0.652x |
| 10K | 16 | 0.551 | 0.018 | 0.030 | 7.251x |
| 10K | 32 | 0.726 | 0.030 | 0.036 | 4.292x |
| 10K | 64 | 0.885 | 0.056 | 0.073 | 2.256x |
| 10K | 128 | 0.964 | 0.109 | 0.147 | 1.167x |
| 25K | 16 | 0.445 | 0.022 | 0.044 | 14.454x |
| 25K | 32 | 0.609 | 0.038 | 0.056 | 8.253x |
| 25K | 64 | 0.782 | 0.067 | 0.096 | 4.723x |
| 25K | 128 | 0.892 | 0.132 | 0.220 | 2.380x |
| 50K | 16 | 0.350 | 0.035 | 0.070 | 21.164x |
| 50K | 32 | 0.501 | 0.053 | 0.083 | 13.832x |
| 50K | 64 | 0.673 | 0.100 | 0.145 | 7.396x |
| 50K | 128 | 0.810 | 0.187 | 0.283 | 3.946x |
| 50K | 256 | 0.920 | 0.370 | 0.591 | 1.995x |
| 50K | 512 | 0.970 | 0.686 | 1.040 | 1.077x |
| 100K | 16 | 0.289 | 0.048 | 0.100 | 35.587x |
| 100K | 32 | 0.431 | 0.073 | 0.111 | 23.366x |
| 100K | 64 | 0.580 | 0.134 | 0.188 | 12.792x |
| 100K | 128 | 0.742 | 0.252 | 0.331 | 6.797x |
| 100K | 256 | 0.871 | 0.501 | 0.655 | 3.421x |
| 100K | 512 | 0.942 | 0.998 | 1.440 | 1.718x |

## Additional 25K construction experiments

| Configuration | Build (ms) | Recall@10 at ef=128 | p50 at ef=128 (ms) |
|---|---:|---:|---:|
| ef_construction=200 | 5,292.614 | 0.912 | 0.149 |
| heuristic disabled | 2,646.229 | 0.875 | 0.139 |

The 25K default-configuration build time was 3,163.242 ms. The alternate construction settings are not direct production recommendations; evaluate on representative image embeddings before choosing defaults.

## Limitations

- Synthetic 64-dimensional vectors are not representative of CLIP image embeddings in every respect.
- Recall and latency should be remeasured with real application embeddings and realistic query distributions.
- These measurements are local benchmark results, not a guarantee of Streamlit Cloud capacity or performance.
