n = 124

| Method | MRR [95% CI] |
|---|---|
| lexical_bm25 | 0.360 [0.295, 0.428] |
| graph_2hop | 0.303 [0.232, 0.376] |
| ppr | 0.300 [0.235, 0.368] |
| rrf_bm25_graph | 0.386 [0.318, 0.455] |
| rrf_bm25_ppr | 0.393 [0.325, 0.460] |

| Comparison | mean diff [95% CI] | P(diff<=0) |
|---|---|---|
| ppr minus graph_2hop | -0.002 [-0.032, +0.025] | 0.555 |
| rrf_bm25_ppr minus lexical_bm25 | +0.033 [-0.023, +0.090] | 0.126 |
| rrf_bm25_ppr minus rrf_bm25_graph | +0.007 [-0.017, +0.033] | 0.282 |
