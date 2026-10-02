# OpenVINO in Studio: convert and chat (Intel Arc / Core Ultra)

The **Convert** page in the Studio sidebar turns a downloaded model into an OpenVINO IR with INT4 or
INT8 weights. On an Intel GPU, the INT variant runs much faster than the original safetensors weights.

It is a thin UI over `model.save_pretrained_openvino(...)` (`unsloth/save.py`). Anything the page
does can also be done from Python:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen3-4B", load_in_4bit = False)
model.save_pretrained_openvino("unsloth/Qwen3-4B-ov_int4", tokenizer = tokenizer, quantization_type = "int4")
```

## Using the page

1. Download a model (Hub or model picker).
2. Open **Convert** in the sidebar, pick the model and a target format:

   | format | what you get |
   |---|---|
   | `ov_int4` | symmetric INT4, group size 128. Fastest on Core Ultra / Arc |
   | `ov_int8` | symmetric INT8 (NNCF). Closer to the original quality |

3. Press **Start Conversion**. A progress bar shows the current stage:
   *Starting → Loading model → Quantizing and saving*. A toast reports success or failure, and the
   model list refreshes.

The result is written next to the source as `<model_id>-ov_int4` or `<model_id>-ov_int8`. It then
appears in the model picker under **Converted**.

## INT recommendation on Intel GPUs

When Studio runs on an Intel GPU (backend `xpu`) and a model is on disk **both** as the original and
as a conversion, the **On Device** and **Converted** tabs show a banner:

> Recommended for your Intel GPU: `org/model-ov_int4` (faster than the unconverted weights)

INT4 wins over INT8 for the same source. Recommended rows sort first in **Converted**. No banner
appears on other backends, or when only the conversion is present. The logic lives in
`intelIntRecommendations()` in
`frontend/src/features/model-picker/components/model-selector/recommended-fit.ts`.

## Chatting with an OpenVINO model

Studio loads an OpenVINO IR directory the same way as any other model: pick it in the model picker
(**Converted** tab), or from the CLI:

```bash
unsloth studio run --model unsloth/ornith-35b-uncensored-int4-ov --port 8000
```

A model counts as OpenVINO when its directory (a local path, `openvino:<path>`, or a repo id in the
HF cache) contains `openvino_model.xml` or `openvino_language_model.xml`. Studio then starts a
private sidecar (`backend/core/inference/openvino_sidecar.py`, OpenVINO GenAI on `GPU`) and proxies
`/v1/chat/completions` to it, like the AMD NPU backend. Loading any other model stops the sidecar.

The sidecar needs `openvino-genai`. Either install it in Studio's environment
(`pip install openvino-genai`) or point Studio at a Python that has it:

```bash
UNSLOTH_OPENVINO_PYTHON=~/openvino-env/.venv/bin/python unsloth studio run --model <ir-dir-or-repo>
```

What the sidecar supports:

| feature | status |
|---|---|
| streaming and plain JSON replies | yes |
| reasoning (`<think>`) | returned as `reasoning_content`; `enable_thinking` switches it off |
| tools | yes. `tools` go into the chat template; `<tool_call>` blocks (Qwen XML or JSON) come back as OpenAI `tool_calls` |
| images / audio / video | refused with 400 |
| `usage` | prompt and completion tokens, in the reply and in the last stream chunk |
| context limit | the model's `max_position_embeddings`, capped by the KV cache (`--cache-gb`, default 3 GB); a longer prompt is a 400, `max_tokens` is cut to what is left |

### Using it from opencode

```jsonc
// ~/.config/opencode/opencode.jsonc
"ornith-local": {
  "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "<key printed by studio run>" },
  "models": { "ornith-35b": { "name": "Ornith 35B (OpenVINO INT4)", "tool_call": true, "reasoning": true } }
}
```

`studio run` reuses the same key across runs (`--api-key-name`, default `cli`), so it only has to
be pasted in once. Then run `opencode --model ornith-local/ornith-35b`.

Without `"tool_call": true` opencode sends no tools, so MCP servers and skills stay unused. Each tool
schema is part of every prompt: a few dozen tools are fine, but hundreds (for example a whole MCP
gateway) make the first reply take minutes on an Arc GPU. Expose a smaller set instead, such as an
[MCPJungle](https://github.com/mcpjungle/MCPJungle) tool group
(`http://127.0.0.1:8080/v0/groups/<group>/mcp`).

### Using it from pi

Add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "unsloth-local": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "<key printed by studio run>",
      "models": [
        { "id": "unsloth/ornith-35b-uncensored-int4-ov", "name": "Ornith 35B INT4 (OpenVINO)", "context": 32768 }
      ]
    }
  }
}
```

```bash
pi --model unsloth-local/unsloth/ornith-35b-uncensored-int4-ov              # interactive
pi --model unsloth-local/unsloth/ornith-35b-uncensored-int4-ov --print "…"  # one-shot
```

### Quick check with curl

```bash
KEY=<key printed by studio run>
curl http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Capital of France? One word."}]}'
# -> choices[0].message.content = "Paris", the reasoning in reasoning_content
# add "stream": true for SSE, or "enable_thinking": false to skip reasoning
```

If port 8000 is already taken (for example by another Studio), `studio run` moves to the next free
port and prints it. Point the client at that port, or stop the other process first.

### Context limit and the KV cache

OpenVINO generates nothing, and reports a normal stop, when a request does not fit the KV cache. The
sidecar estimates the cache's capacity from the model config (f16 K+V with a 15% margin) and turns
such requests into a 400 up front. On an Arc Pro B60 with Ornith 35B INT4 the 3 GB cache holds about
133k tokens, far below the model's 262k, and the card has no memory left to grow it. Set the
client's context limit to match (opencode: `"limit": {"context": 130000}`), so it compacts the
history in time.

### Integration tests

`tests/test_openvino_live.py` drives a running Studio over HTTP (tool calls, reasoning, validation,
token limits, disconnects):

```bash
UNSLOTH_E2E_OPENVINO=1 UNSLOTH_E2E_BASE_URL=http://127.0.0.1:8000 UNSLOTH_E2E_API_KEY=<key> \
  pytest studio/backend/tests/test_openvino_live.py
# UNSLOTH_E2E_OPENVINO_LONG=1 adds the context-boundary case (minutes of prefill)
```

### How tool calls work

The sidecar passes `tools` to the model's chat template and holds back streamed text from the first
`<tool_call>` on. At the end of the reply it parses the blocks into OpenAI `tool_calls` (parameter
types follow the tool's JSON schema) and finishes with `finish_reason: "tool_calls"`. Markup it
cannot parse is returned as plain content. Assistant `tool_calls` and `role: "tool"` messages are
kept in the history, and several system messages (opencode sends two) are joined into one, since
Qwen-style templates accept a single leading system message.

## API

| method | path | body / query | result |
|---|---|---|---|
| `POST` | `/api/convert` | `{"model_id": "...", "format": "ov_int4" \| "ov_int8"}` | `200` started, `400` unknown format, `409` already running |
| `GET` | `/api/convert/status` | `?model_id=...` | `{"state": "running" \| "done" \| "error", "stage": "..."}`, `404` if never started |

The conversion runs in a separate Python process (`sys.executable`), so it neither blocks the server
nor fights its GPU state. The full log is written to `$TMPDIR/convert_<model_id>.log`. When a
conversion fails, `stage` carries that path.

## Limits

- Conversion job state is kept in memory, so a server restart forgets running and finished jobs. The files on
  disk are unaffected.
- `save_pretrained_openvino` reports no percentage, so the progress bar is indeterminate and shows
  only the stage.
