# Colab T4 behavior test (live status)

Notebook: https://colab.research.google.com/drive/1HA-AoQofoZewbI3k8pfq7UohxpbgAoEV (Untitled1.ipynb, T4, Python 3.13)
Purpose: serve E4B on a T4 and check prompt and tool-call behavior. This gives no benchmark score.

## Cells
1. Install check: `pip -q install vllm` (earlier run printed vLLM 0.33.0).
2. Launch (runs in background, log at /content/vllm.log):
   `vllm serve google/gemma-4-E4B-it-qat-w4a16-ct --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --port 8000`
3. Next: poll http://localhost:8000/health, then send a chat request with tools and the new prompt.

## Log (newest last)
- 2026-11-07 18:27 UTC+7: runtime had reset; reinstalling vLLM, then launching the server.
- 2026-11-07 ~18:35: vLLM 0.33.0 installed on Python 3.13 (works on T4).
- 2026-11-07 ~18:40: server launched; model loaded in 4.7 s (Marlin kernel, TRITON_ATTN backend), now compiling and capturing CUDA graphs; health not yet up.
- 2026-11-07 ~18:45: tool-call test cell is waiting for /health (timeout 500 s), then sends one chat request with 3 tools.

- 2026-11-07 ~18:50: default and eager runs both failed on T4 with a Triton OutOfResources (shared memory) error; Gemma 4 has 512-wide full-attention heads and the TRITON_ATTN backend needs more shared memory than the T4 has.
- 2026-11-07 ~19:00: FlexAttention relaunch reached health, but the tool-call request was not run before Colab disconnected for inactivity (19:54). Needs a reconnect and a fresh install.

- 2026-11-07 ~20:35: CLOSED, inconclusive. vLLM on T4: FlexAttention rejects KV sharing; Triton attention needs 98 KB shared memory (T4 max 64 KB) even with block size 16, so every request returns HTTP 500. Transformers fallback: the w4a16 model decompresses to about 13.5 GB fp16 and runs out of memory on the 15 GB T4 (CPU offload fails on meta tensors). No model behavior was observed. Runtime deleted.
- Next option if a behavior test is still wanted: a GPU with sm80 or newer (L4 or A100), which matches the scoring hardware (4x L4).

## Authorization
Restarting or deleting the Colab runtime is authorized by the user (2026-11-07).
