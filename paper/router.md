| Method | MRR | 95% CI |
|---|---|---|
| lexical_bm25 | 0.360 | [0.295, 0.428] |
| seed_only | 0.254 | [0.187, 0.325] |
| graph_2hop | 0.303 | [0.232, 0.376] |
| embedding_seed | 0.202 | [0.149, 0.260] |
| hybrid_rrf | 0.377 | [0.310, 0.446] |
| hybrid_bm25_graph | 0.386 | [0.318, 0.455] |
| router_seed_rule | 0.386 | [0.318, 0.455] |
| oracle_best_of_all | 0.544 | [0.473, 0.616] |
| oracle_bm25_or_graph | 0.465 | [0.395, 0.535] |

Router minus BM25 and router minus the fixed fusion (paired bootstrap):

| Comparison | mean diff | 95% CI | P(diff<=0) |
|---|---|---|---|
| router_seed_rule minus lexical_bm25 | +0.026 | [-0.034, +0.086] | 0.199 |
| router_seed_rule minus hybrid_bm25_graph | +0.000 | [+0.000, +0.000] | 1.000 |
