# Local Decision API: Laya and Clef

Enable **Settings → API → Decision API** and select a model. Laya multilingual remains the default. Clef Flash (9B) and Clef (27B) use the official joint-schema head, not chat generation. Downloads pin the backbone and head revisions; model-repository Python is never executed.

| Backend | Text / JSON | Images | Video / audio |
|---|---|---|---|
| Laya | Yes | Rejected | Rejected |
| Clef · PyTorch | Yes | PNG, JPEG, WebP | Rejected |
| Clef · llama.cpp integration | Yes | Use PyTorch | Rejected |
| Saved Decisions connection | Yes | Rejected | Rejected |

**Auto** selects native text decisions when the installed executable supports `/v1/systemone`; otherwise it reports its PyTorch selection. The latest Unsloth llama.cpp release verified here, `b11443-mix-d65395f` (source `4021fda1a7d2699904843d45e3267a66e4c3af2b`), includes that endpoint and Clef image-projector conversion. This integration's pinned Q8 download plan does not yet include a projector, so images still select PyTorch. Empty instructions and single-level scores also select PyTorch in Auto; forced native mode rejects them. Startup and inference errors never trigger a retry on another backend.

Settings displays the selected runtime and reason. Responses identify the runtime in `x-unsloth-decision-backend`. Older installed binaries may need updating; the installer defaults to the latest usable Unsloth release.

## Resources and lifecycle

PyTorch downloads are about 19 / 55 GB. Allow roughly 20 / 56 GB GPU memory, or 40 / 112 GB CPU RAM for float32 execution plus input overhead. CPU is the default and can be slow. GPU mode uses one selected device; unload resident chat/image/video models first. Native Q8 downloads are about 10 / 29 GB.

Weights must finish downloading before loading. Startup can return retryable `503 model_loading`; retry after `Retry-After`. Unload, shutdown and five minutes of inactivity retire the owned worker. PyTorch refuses requests exceeding 16,384 tokens; native requests have a 2,048-token limit. Owner fine-tunes retain the existing GPU-only Unsloth runtime and account-isolation rules.

## Requests and SDKs

Use `POST /v1/systemone` with the usual Unsloth authentication policy. Settings writes and downloads belong to the owner. Keyless callers can use only the configured model. Existing default aliases continue to work; authenticated callers can explicitly select `clef-flash`, `clef`, their Cloudflare IDs, or an accessible fine-tune.

The response contains `model`, `answers`, `usage` and `x-typesafe-request-id`. Choice and score answers include probabilities; noul returns the probability of true. No output tokens are generated. This is System One compatibility, not an invented OpenAI `/v1/decisions` schema.

Python's TypeSafe SDK accepts images through `extra_body`, without a fork:

```python
import base64
import os
from pathlib import Path
from typesafe_sdk import Choice, TypeSafeClient

image = "data:image/png;base64," + base64.b64encode(Path("image.png").read_bytes()).decode()
with TypeSafeClient(base_url="http://127.0.0.1:8888", api_key=os.environ["STUDIO_API_KEY"]) as client:
    result = client.system_one(
        state="Inspect the image.",
        questions={"color": Choice(instructions="Dominant color?", criteria={"red": "Red", "blue": "Blue"})},
        extra_body={"images": [image]},
    )
    print(result.choices["color"].choice)
```

The JavaScript SDK forwards extra properties from a typed request variable:

```typescript
import { choice, TypeSafeClient, type SystemOneRequest } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({ baseURL: "http://127.0.0.1:8888", apiKey });
const questions = { color: choice("Dominant color?", { red: "Red", blue: "Blue" }) };
const request: SystemOneRequest<typeof questions> & { images: string[] } = {
  state: "Inspect the image.", questions, images: [image],
};
const result = await client.systemOne(request);
console.log(result.answers.color.choice);
```

Use your actual server port and API key. Images are base64 data URLs only; remote URLs are never fetched. Limits: four images, 4 MiB and 16 million pixels per image, 8 MiB decoded total, and a 13 MiB request body. MIME must match PNG/JPEG/WebP content; animated and malformed images are rejected. `image_url` parts in state message arrays follow the same rules. Unknown fields, processor overrides, video and audio are rejected rather than ignored.

Sources: [Clef Flash](https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6), [Clef](https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c), [latest verified native release](https://github.com/unslothai/llama.cpp/releases/tag/b11443-mix-d65395f). Native image support was checked in source, not revalidated with live model inference for this refresh.
