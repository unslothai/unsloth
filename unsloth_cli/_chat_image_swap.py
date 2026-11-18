# SPDX-License-Identifier: AGPL-3.0-only
"""Run a client-side image tool between two completed text inference requests."""

import json


IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate one image. Give the image generator a complete visual prompt.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
}


def image_tool_turn(api, body, text_id, image_load, on_image):
    """Return a plain reply or the assistant call and matching tool result for continuation."""
    first = api(
        "/v1/chat/completions",
        {
            **body,
            "model": text_id,
            "stream": False,
            "tools": [IMAGE_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "enable_tools": False,
        },
    )
    choice = first["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        return choice, []
    if len(calls) != 1 or calls[0]["function"]["name"] != "generate_image":
        raise ValueError("Expected a single generate_image call.")
    call = calls[0]
    prompt = json.loads(call["function"]["arguments"]).get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or not call.get("id"):
        raise ValueError("The image call needs a prompt and a tool-call ID.")
    response = api(
        "/api/inference/images/chat-swap",
        {
            "text_id": text_id,
            "image_load": image_load,
            "prompt": prompt,
        },
    )
    generated = response["images"][0]
    image = {"gallery_image_id": generated["id"], "url": generated["url"]}
    on_image(image)
    result = {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(image)}
    return choice, [message, result]
