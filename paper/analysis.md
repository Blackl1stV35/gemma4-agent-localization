n = 124 tasks, 10000 bootstrap resamples (seed 0)

| Method | MRR | 95% CI | hit@10 | 95% CI |
|---|---|---|---|---|
| lexical_bm25 | 0.360 | [0.295, 0.428] | 0.629 | [0.540, 0.710] |
| seed_only | 0.254 | [0.187, 0.325] | 0.355 | [0.274, 0.435] |
| graph_2hop | 0.303 | [0.232, 0.376] | 0.411 | [0.331, 0.500] |
| embedding_seed | 0.202 | [0.149, 0.260] | 0.403 | [0.322, 0.492] |
| hybrid_rrf | 0.377 | [0.310, 0.446] | 0.661 | [0.573, 0.742] |
| hybrid_bm25_graph | 0.386 | [0.318, 0.455] | 0.677 | [0.597, 0.758] |

Paired differences in MRR (bootstrap over tasks):

| Comparison | mean diff | 95% CI | P(diff<=0) |
|---|---|---|---|
| hybrid_bm25_graph minus lexical_bm25 | +0.026 | [-0.034, +0.086] | 0.199 |
| hybrid_rrf minus lexical_bm25 | +0.017 | [-0.046, +0.081] | 0.294 |
| graph_2hop minus seed_only | +0.049 | [+0.017, +0.082] | 0.001 |
| embedding_seed minus graph_2hop | -0.100 | [-0.167, -0.033] | 1.000 |
| hybrid_bm25_graph minus hybrid_rrf | +0.009 | [-0.029, +0.046] | 0.331 |

Per-repository MRR:

| Repository | n | lexical_bm25 | seed_only | graph_2hop | embedding_seed | hybrid_rrf | hybrid_bm25_graph |
|---|---|---|---|---|---|---|---|
| fastapi | 64 | 0.389 | 0.150 | 0.218 | 0.151 | 0.343 | 0.340 |
| httpx | 1 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| requests | 13 | 0.666 | 0.387 | 0.511 | 0.442 | 0.645 | 0.579 |
| rich | 46 | 0.218 | 0.355 | 0.347 | 0.189 | 0.336 | 0.381 |
