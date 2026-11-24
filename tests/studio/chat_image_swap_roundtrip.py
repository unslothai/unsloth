# SPDX-License-Identifier: AGPL-3.0-only

"""Exercise one automatic image tool call against a dedicated local Unsloth instance.

Supply the same JSON load requests used by the text and image model pickers.
The text request can select either a GGUF or a Transformers model. The
UNSLOTH_TOKEN environment variable holds the local session's bearer token.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unsloth_cli._inference import read_json_checking_deferred_error, urlopen_no_redirect


def roundtrip(api, text_load, image_load, question):
    from unsloth_cli._chat_image_swap import image_tool_turn

    model = api("/api/inference/load", text_load)["model"]
    messages = [{"role": "user", "content": question}]
    body = {"model": model, "messages": messages, "enable_tools": False}
    _, history = image_tool_turn(
        api, body, model, image_load, lambda image: print("Saved", image["url"], flush = True)
    )
    if not history:
        raise RuntimeError("The text model did not call generate_image.")
    final = api(
        "/v1/chat/completions", {**body, "messages": [*messages, *history], "tool_choice": "none"}
    )
    return final["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--text-load", type = Path, required = True)
    parser.add_argument("--image-load", type = Path, required = True)
    parser.add_argument("--port", type = int, default = 8888)
    parser.add_argument("prompt")
    args = parser.parse_args()
    token = os.environ["UNSLOTH_TOKEN"]

    def api(path, body = None):
        request = Request(
            f"http://127.0.0.1:{args.port}{path}",
            data = json.dumps(body).encode() if body is not None else None,
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        return read_json_checking_deferred_error(
            request.full_url, urlopen_no_redirect(request, timeout = 1200)
        )

    answer = roundtrip(
        api,
        json.loads(args.text_load.read_text()),
        json.loads(args.image_load.read_text()),
        args.prompt,
    )
    print(answer)


if __name__ == "__main__":
    main()
