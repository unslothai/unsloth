"""Voice mode on a Studio Decision fine-tune: judge, filler and interject as one checkpoint's questions.

The states and questions here are the training side's (voice-endpointing-data
decision/build_decision_v2.py); a change on either side means a retrain. Picked by
``UNSLOTH_VOICE_DECISION_CHECKPOINT`` (a Decision catalog name such as ``laya-ft:<run folder>``).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

ENV_CHECKPOINT = "UNSLOTH_VOICE_DECISION_CHECKPOINT"

TURN_DONE = {
    "turn_done": {
        "type": "noul",
        "instructions": (
            "The state is what a person has said so far to a voice assistant, as speech "
            "recognition wrote it. Have they finished what they wanted to say, so the "
            "assistant should answer now, rather than paused mid-thought?"
        ),
    }
}

INTERJECT = {
    "interject": {
        "type": "choice",
        "instructions": (
            "A voice assistant is still thinking about the request and has not answered yet. "
            "The person spoke again. What should the assistant do with what they said?"
        ),
        "criteria": {
            "amend": "it adds to or adjusts the request: keep the request, fold this in and start over",
            "replace": "it cancels the request or changes direction: drop the request and start over with only this",
            "queue": "it is a separate request: hold it and answer it after this reply",
            "ignore": "a backchannel, self-talk or someone else in the room: nothing to act on",
        },
    }
}

FILLER_INSTRUCTIONS = (
    "The state is a finished request to a voice assistant, as speech recognition "
    "wrote it. The assistant may say one short line aloud while it thinks. Which "
    "kind of line fits, or none?"
)

# The app's own action set; amend and replace are both a restart, told apart by ``kind``.
ACTION_OF = {"amend": "restart", "replace": "restart", "queue": "queue", "ignore": "ignore"}


def checkpoint_name() -> Optional[str]:
    return (os.environ.get(ENV_CHECKPOINT) or "").strip() or None


def filler_question(bank: dict) -> dict:
    return {
        "filler": {
            "type": "choice",
            "instructions": FILLER_INSTRUCTIONS,
            "criteria": {cls: v["desc"] for cls, v in bank["classes"].items()},
        }
    }


def judge_state(text: str) -> str:
    return text


def filler_state(text: str, prev_q: bool) -> str:
    return f"(the assistant just asked them a question)\n{text}" if prev_q else text


def interject_state(prompt: str, utterance: str, elapsed_s: float) -> str:
    when = f" ({elapsed_s:.0f} seconds into thinking)" if elapsed_s and elapsed_s > 0 else ""
    return f"Request: {prompt}\nSpoken over the thinking{when}: {utterance}"


class DecisionVoice:
    """One resident Decision checkpoint answering all three voice questions."""

    def __init__(
        self,
        name: str,
        decide = None,
    ):
        from core.systemone import catalog, laya_runtime

        checkpoint = catalog.resolve(name)
        if checkpoint is None or not hasattr(checkpoint, "layout"):
            raise RuntimeError(f"{ENV_CHECKPOINT}={name!r} is not a Decision checkpoint")
        self.checkpoint = checkpoint
        self.backend = f"decision:{checkpoint.layout}"
        self._decide = decide or laya_runtime.decide

    def ask(self, state: str, question: dict) -> dict:
        from core.systemone.laya_runtime import Unavailable

        name = next(iter(question))
        try:
            result = self._decide(self.checkpoint, state, question)
        except Unavailable as exc:
            raise RuntimeError(f"Decision model unavailable: {exc.message}") from exc
        if result.get("truncated"):
            raise RuntimeError("Decision state exceeds the checkpoint's context window")
        return result["answers"][name]

    def judge(self, text: str) -> float:
        return float(self.ask(judge_state(text), TURN_DONE)["noul"])

    def classify(self, text: str, prev_q: bool, question: dict) -> tuple[str, float]:
        a = self.ask(filler_state(text, prev_q), question)
        return str(a["choice"]), float(
            a["probabilities"].get(a["choice"], a.get("confidence", 0.0))
        )

    def interject(self, prompt: str, utterance: str, elapsed_s: float) -> dict:
        a = self.ask(interject_state(prompt, utterance, elapsed_s), INTERJECT)
        p = {k: float(v) for k, v in a["probabilities"].items()}
        kind = str(a["choice"])
        action = ACTION_OF.get(kind, "restart")
        probs = {
            "restart": round(p.get("amend", 0.0) + p.get("replace", 0.0), 4),
            "ignore": round(p.get("ignore", 0.0), 4),
            "queue": round(p.get("queue", 0.0), 4),
        }
        return {
            "action": action,
            "kind": kind,
            "confidence": round(p.get(kind, 0.0), 4),
            "probs": probs,
            "merged_prompt": (
                utterance
                if kind == "replace"
                else f"{prompt}\n{utterance}"
                if kind == "amend"
                else None
            ),
        }


class DecisionJudge:
    def __init__(self, voice: DecisionVoice):
        self.voice, self.backend = voice, voice.backend

    def __call__(self, text: str) -> float:
        return self.voice.judge(text)


class DecisionFillerClassifier:
    """Drop-in for the ONNX/TF-IDF classifier inside ``_Filler``."""

    def __init__(self, voice: DecisionVoice, bank: dict):
        self.voice, self.question = voice, filler_question(bank)

    def classify(
        self,
        text: str,
        prev_q: bool = False,
    ) -> tuple[str, float]:
        return self.voice.classify(text, prev_q, self.question)


class DecisionInterject:
    def __init__(self, voice: DecisionVoice):
        self.voice, self.backend = voice, voice.backend

    def predict(
        self,
        prompt: str,
        utterance: str,
        elapsed_s: float = 2.0,
    ) -> dict:
        t0 = time.perf_counter()
        out = self.voice.interject(prompt, utterance, elapsed_s)
        out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out
