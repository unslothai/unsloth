import os, sys

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

from models.inference import ChatMessage
from routes.inference import _openai_messages_for_passthrough


_TINY = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


class _P:
    def __init__(self, messages, image_base64 = None):
        self.messages = messages
        self.image_base64 = image_base64


def _img_url_part(b64 = _TINY):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def test_prior_user_image_does_not_block_new_base64():
    p = _P(
        messages = [
            ChatMessage(
                role = "user", content = [{"type": "text", "text": "q1"}, _img_url_part()]
            ),
            ChatMessage(role = "assistant", content = "a1"),
            ChatMessage(role = "user", content = "q2 no inline"),
        ],
        image_base64 = _TINY,
    )
    out = _openai_messages_for_passthrough(p)
    # last user must receive spliced image
    assert isinstance(out[-1]["content"], list)
    assert any(x.get("type") == "image_url" for x in out[-1]["content"])


def test_last_user_with_inline_image_skips_splice():
    p = _P(
        messages = [
            ChatMessage(role = "user", content = "earlier"),
            ChatMessage(
                role = "user", content = [{"type": "text", "text": "last"}, _img_url_part()]
            ),
        ],
        image_base64 = _TINY,
    )
    out = _openai_messages_for_passthrough(p)
    last_parts = out[-1]["content"]
    assert sum(1 for x in last_parts if x.get("type") == "image_url") == 1


def test_no_user_messages_appends_trailing_user():
    p = _P(
        messages = [ChatMessage(role = "system", content = "sys")],
        image_base64 = _TINY,
    )
    out = _openai_messages_for_passthrough(p)
    assert out[-1]["role"] == "user"
    assert any(x.get("type") == "image_url" for x in out[-1]["content"])


def test_three_user_turns_only_last_receives_splice():
    p = _P(
        messages = [
            ChatMessage(role = "user", content = "u1"),
            ChatMessage(role = "assistant", content = "a1"),
            ChatMessage(role = "user", content = "u2"),
            ChatMessage(role = "assistant", content = "a2"),
            ChatMessage(role = "user", content = "u3"),
        ],
        image_base64 = _TINY,
    )
    out = _openai_messages_for_passthrough(p)
    assert isinstance(out[0]["content"], str)
    assert isinstance(out[2]["content"], str)
    assert isinstance(out[4]["content"], list)
