# OpenVINO in Studio (Intel Arc / Core Ultra)

## Convert

**Convert** (sidebar, under More) turns a downloaded model into an OpenVINO IR with `ov_int4` or
`ov_int8` weights. It runs `model.save_pretrained_openvino(...)` in a separate process and writes
`exports/<model>-openvino/<format>`, with the log beside it as `convert_<format>.log`. The same
from Python:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen3-4B", load_in_4bit = False)
model.save_pretrained_openvino("Qwen3-4B-ov_int4", tokenizer = tokenizer, quantization_type = "int4")
```

API (owner only): `POST /api/convert {"model_id", "format"}` (400 unknown format, 409 already
running), `GET /api/convert/status?model_id=` -> `{"state": "running" | "done" | "error", "stage"}`.
Job state lives in memory and is lost on restart.

## Chat

A model is served by OpenVINO when its directory (local path, `openvino:<path>`, or a repo in the HF
cache) holds `openvino_model.xml` or `openvino_language_model.xml`. A directory that also ships
safetensors loads the standard way unless OpenVINO GenAI is available. Studio then starts a private
OpenVINO GenAI server on the `GPU` device and proxies `/v1/chat/completions` to it, like the AMD NPU
backend; loading any other model stops it.

It needs `openvino-genai` in Studio's Python, or a Python that has it:

```bash
UNSLOTH_OPENVINO_PYTHON=~/ov/.venv/bin/python unsloth studio run --model unsloth/ornith-35b-uncensored-int4-ov
```

- Reasoning comes back as `reasoning_content`; `enable_thinking: false` turns it off.
- `tools` go into the chat template; `<tool_call>` blocks (Qwen XML or JSON) come back as OpenAI
  `tool_calls`. Clients such as opencode need `"tool_call": true` on the model to send tools.
- Images, audio and video are refused with 400.
- `usage` is in the reply and in the last stream chunk.
- Context = `max_position_embeddings` capped by an estimate of the 3 GB KV cache (about 133k tokens
  for Ornith 35B INT4 on an Arc Pro B60). A longer prompt is a 400 (OpenVINO would otherwise return
  an empty stop, openvino.genai#4545), and `max_tokens` is cut to what is left. Set the client's
  context limit to match.
