# OpenVINO Convert (Intel Arc / Core Ultra)

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

## API

| method | path | body / query | result |
|---|---|---|---|
| `POST` | `/api/convert` | `{"model_id": "...", "format": "ov_int4" \| "ov_int8"}` | `200` started, `400` unknown format, `409` already running |
| `GET` | `/api/convert/status` | `?model_id=...` | `{"state": "running" \| "done" \| "error", "stage": "..."}`, `404` if never started |

The conversion runs in a separate Python process (`sys.executable`), so it neither blocks the server
nor fights its GPU state. The full log is written to `$TMPDIR/convert_<model_id>.log`. When a
conversion fails, `stage` carries that path.

## Limits

- Job state is kept in memory, so a server restart forgets running and finished jobs. The files on
  disk are unaffected.
- `save_pretrained_openvino` reports no percentage, so the progress bar is indeterminate and shows
  only the stage.
