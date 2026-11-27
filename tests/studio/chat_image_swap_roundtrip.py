# SPDX-License-Identifier: AGPL-3.0-only

"""Exercise two consecutive image tool calls against a dedicated local Unsloth instance.

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
    model = api("/api/inference/load", text_load)["model"]
    messages = []
    seen = {image["id"] for image in api("/api/inference/images/gallery")["images"]}
    for tools in (["generate_image", "web_search"], ["generate_image"]):
        prompt = (
            question
            if not messages
            else "Generate a new image of a blue sailboat using the image tool."
        )
        messages.append({"role": "user", "content": prompt})
        response = api(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "enable_tools": True,
                "enabled_tools": tools,
                "image_generator": image_load["model_path"],
                "max_tool_calls_per_message": 1,
                "parallel_tool_calls": False,
                "permission_mode": "off",
                "tool_choice": {"type": "function", "function": {"name": "generate_image"}},
            },
        )
        current = {image["id"] for image in api("/api/inference/images/gallery")["images"]}
        if len(current - seen) != 1:
            raise RuntimeError(f"Expected one new gallery image with tools {tools}.")
        seen = current
        messages.append(response["choices"][0]["message"])
        if not messages[-1]["content"]:
            raise RuntimeError("Chat did not continue after the image tool.")
    return messages[-1]["content"]


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
