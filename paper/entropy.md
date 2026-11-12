n = 124

| Method | MRR [95% CI] |
|---|---|
| lexical_bm25 | 0.360 [0.295, 0.428] |
| bits_binary | 0.287 [0.234, 0.344] |
| bits_sat | 0.289 [0.235, 0.348] |
| conc | 0.284 [0.230, 0.339] |
| cov | 0.287 [0.234, 0.344] |
| rrf_bm25_bits | 0.333 [0.272, 0.395] |
| rrf_bm25_conc | 0.341 [0.281, 0.403] |
| rrf_bm25_cov | 0.333 [0.272, 0.395] |
| rrf_bm25_graph_conc | 0.394 [0.328, 0.461] |

| Comparison | mean diff [95% CI] | P(diff<=0) |
|---|---|---|
| bits_binary minus lexical_bm25 | -0.072 [-0.138, -0.007] | 0.986 |
| bits_sat minus lexical_bm25 | -0.071 [-0.139, -0.003] | 0.980 |
| conc minus lexical_bm25 | -0.076 [-0.140, -0.015] | 0.992 |
| cov minus lexical_bm25 | -0.072 [-0.138, -0.007] | 0.986 |
| rrf_bm25_bits minus lexical_bm25 | -0.027 [-0.070, +0.014] | 0.903 |
| rrf_bm25_conc minus lexical_bm25 | -0.019 [-0.066, +0.025] | 0.792 |
| rrf_bm25_cov minus lexical_bm25 | -0.027 [-0.070, +0.014] | 0.903 |
| rrf_bm25_graph_conc minus lexical_bm25 | +0.034 [-0.027, +0.097] | 0.141 |
