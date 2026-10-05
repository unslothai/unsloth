# Local Decision API: Laya and Clef

Enable **Settings → API → Decision API** and select the model. Laya multilingual stays the recommended default. **Clef Flash (9B)** and **Clef (27B)** score your allowed answers with the official joint-schema head, not chat generation. The first Clef selection offers a revision-pinned download, including the vision backbone and separate head. No model-repository Python is executed; Studio ships the reviewed adapter.

| Studio backend | Text / JSON | Images | Video | Audio |
|---|---|---|---|---|
| Laya (default) | Yes | Rejected | Rejected | Rejected |
| Clef Flash / Clef · PyTorch | Yes | PNG, JPEG, WebP | Rejected | Rejected |
| Clef Flash / Clef · compatible llama.cpp | Yes | Rejected (no projector) | Rejected | Rejected |
| Saved Decisions connection | Yes | Rejected by Studio | Rejected | Rejected |

Clef's source model accepts video frame arrays, but Studio does **not** expose a video decoder/frame contract. Use a reviewed frame-extraction workflow to submit up to four still images, without claiming temporal video understanding. Audio has no supported model path.

**Runtime → Auto** prefers llama.cpp when the configured executable contains `/v1/systemone`; the current Unsloth bundle (`b11160-mix-a6922cc`) does not, so Auto explicitly reports its PyTorch selection. Compatible native builds use pinned Q8 GGUFs retaining the joint head. Native Clef has no vision projector: images, empty instructions and single-level scores use PyTorch in Auto, or are refused in forced llama.cpp mode. Choose **PyTorch** to preload one backend for text and images. Native startup, authentication and inference errors propagate; they never trigger a backend retry. Settings shows the selection/reason and resident runtime; HTTP responses identify the actual runtime in `x-unsloth-decision-backend`. Neither path uses base-Qwen chat generation.

## Resources and lifecycle

Downloads are about 19 GB (Flash) and 55 GB (Clef). Allow roughly 20 / 56 GB GPU memory respectively, or 40 / 112 GB CPU RAM with the float32 CPU path plus input/allocator overhead. CPU execution can be very slow. CPU remains the default; GPU uses one selected device, not all devices. Unload the resident chat/image/video model before starting Clef on GPU; a Decisions request does not silently evict it.

Native Q8 downloads are about 10 / 29 GB. In Auto, the image path additionally needs the PyTorch snapshot; select PyTorch in Settings to download it before submitting images.

Weights must finish downloading before a Clef request loads them. Startup can return retryable `503 model_loading`; inspect Settings and retry after `Retry-After`. Failures appear in Settings. **Unload** releases the worker; idle workers unload after five minutes. Changing settings and Studio shutdown retire the owned worker. PyTorch accepts at most 16,384 tokens; the native server is configured for 2,048 and refuses overflow. Neither silently shortens inputs.

## HTTP: System One, not OpenAI Decisions preview

Use `POST http://127.0.0.1:8888/v1/systemone` (your actual Studio port), with the same Studio authentication/key policy as the other API routes. Settings writes/downloads belong to the Studio owner. Keyless callers may use only the configured model. The existing `default`, `laya`, `jev-latest`, `jev-preview` and `openjev-latest` aliases still reach that configured model; explicit `clef-flash`, `clef`, and their `Cloudflare/…` IDs select the corresponding local checkpoint for authenticated callers.

OpenAI has announced a limited-preview finite-answer Decisions API, but has not published a wire schema in the public reference/SDK. This endpoint claims **System One compatibility only**; there is no invented `/v1/decisions` endpoint.

The response retains `model`, `answers`, `usage`, and `x-typesafe-request-id`. `choice` and `score` return per-option probabilities; `noul` returns the probability of true. There are no generated output tokens.

```python
import base64
import os
from pathlib import Path
import httpx

image = "data:image/png;base64," + base64.b64encode(Path("image.png").read_bytes()).decode()
response = httpx.post(
    "http://127.0.0.1:8888/v1/systemone",
    headers={"Authorization": f"Bearer {os.environ['STUDIO_API_KEY']}"},
    json={
        "model": "default",
        "state": "Inspect the attached image.",
        "images": [image],
        "questions": {
            "color": {"type": "choice", "instructions": "What is the dominant color?",
                      "criteria": {"red": "Red", "blue": "Blue", "green": "Green"}}
        },
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["answers"]["color"])
```

Image limits, including images embedded in `state` message content:

- Maximum **4 images**, each **4 MiB decoded** and **16 million pixels**.
- Maximum **8 MiB decoded total**, **13 MiB request body** (including JSON/base64).
- Base64 **data URLs only**; remote URLs are never fetched. MIME must match the actual PNG/JPEG/WebP data. Malformed/animated images are rejected.
- The native `state` extension `[{"role":"user","content":[{"type":"text","text":"Inspect"},{"type":"image_url","image_url":{"url":image}}]}]` follows the same bounds. Images are extracted for the visual encoder; ordinary structured JSON state stays JSON.
- Unknown request fields, unsupported message content types, `videos`, `audio`, and processor overrides are rejected, not ignored. A saved connection's hosted capability is not assumed from its model name.

## Existing TypeSafe SDKs

Text clients do not need changes: point their SDK base URL to Studio's origin (without `/v1`) and use a Studio API key. The SDK itself appends `/v1/systemone`. SDK transport support is distinct from TypeSafe's hosted API, whose public contract documents text request fields only.

Python's official SDK supports the extension without a fork:

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8888", api_key=os.environ["STUDIO_API_KEY"]) as client:
    result = client.system_one(
        state="Inspect the attached image.",
        questions={"color": Choice(instructions="What is the dominant color?",
                                    criteria={"red": "Red", "blue": "Blue", "green": "Green"})},
        extra_body={"images": [image]},
    )
    print(result.choices["color"].choice)
```

The JavaScript SDK forwards extra properties from a request variable. A typed intersection preserves answer inference without an `any` cast or SDK fork:

```typescript
import { choice, TypeSafeClient, type SystemOneRequest } from "@typesafe-ai/sdk";

const apiKey = process.env.STUDIO_API_KEY;
if (!apiKey) throw new Error("Set STUDIO_API_KEY");
const client = new TypeSafeClient({
  baseURL: "http://127.0.0.1:8888",
  apiKey,
});
const questions = {
  color: choice("What is the dominant color?", { red: "Red", blue: "Blue", green: "Green" }),
};
const request: SystemOneRequest<typeof questions> & { images: string[] } = {
  state: "Inspect the attached image.", questions, images: [image],
};
const result = await client.systemOne(request);
console.log(result.answers.color.choice);
```

Sources: [Clef Flash](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6), [Clef](https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c), [llama.cpp Decisions](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp), [Clef converter](https://github.com/ggml-org/llama.cpp/blob/c173a53bdfca1047c710018dc934a6d67a8b010f/conversion/clef.py), [Cloudflare hosted image schema](https://developers.cloudflare.com/workers-ai/models/clef-flash/schema-input.json). Model, native runtime, hosted API and SDK capabilities are intentionally distinguished.
