## Sensitivity (MRR with 95 percent bootstrap CI, n = 124)

| Variant | MRR [95% CI] |
|---|---|
| graph_1hop | 0.296 [0.225, 0.369] |
| graph_2hop | 0.303 [0.232, 0.376] |
| graph_3hop | 0.316 [0.247, 0.387] |
| bm25_graph_rrf_k10 | 0.398 [0.329, 0.466] |
| bm25_graph_rrf_k60 | 0.386 [0.318, 0.455] |
| bm25_graph_rrf_k100 | 0.389 [0.320, 0.460] |

## Learned router (logistic regression, leave-one-out)

| Method | MRR [95% CI] |
|---|---|
| lexical_bm25 | 0.360 [0.295, 0.428] |
| hybrid_bm25_graph | 0.386 [0.318, 0.455] |
| router_loo_with_repo (BM25 vs BM25+graph) | 0.387 [0.320, 0.455] |
| router_loo_no_repo (BM25 vs BM25+graph) | 0.359 [0.294, 0.427] |
| router_loo_with_repo (BM25 vs graph) | 0.384 [0.316, 0.455] |

Paired differences (bootstrap):

| Comparison | mean diff [95% CI] | P(diff<=0) |
|---|---|---|
| router_loo_with_repo (BM25 vs BM25+graph) minus lexical_bm25 | +0.027 [-0.002, +0.059] | 0.032 |
| router_loo_with_repo (BM25 vs BM25+graph) minus hybrid_bm25_graph | +0.001 [-0.052, +0.053] | 0.478 |
| router_loo_no_repo (BM25 vs BM25+graph) minus lexical_bm25 | -0.001 [-0.002, +0.000] | 1.000 |
| router_loo_no_repo (BM25 vs BM25+graph) minus hybrid_bm25_graph | -0.027 [-0.086, +0.034] | 0.805 |
| router_loo_with_repo (BM25 vs graph) minus lexical_bm25 | +0.025 [+0.002, +0.053] | 0.017 |
| router_loo_with_repo (BM25 vs graph) minus hybrid_bm25_graph | -0.001 [-0.060, +0.058] | 0.517 |
