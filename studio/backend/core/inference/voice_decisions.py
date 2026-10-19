"""Voice mode on a Studio Decision fine-tune: judge, filler and interject as one checkpoint's questions.

The states and questions here are the training side's (voice-endpointing-data
decision/build_decision_v2.py); a change on either side means a retrain. Picked by the
``voice_decision_checkpoint`` setting, else ``UNSLOTH_VOICE_DECISION_CHECKPOINT`` (a Decision
catalog name such as ``laya-ft:<run folder>``).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

ENV_CHECKPOINT = "UNSLOTH_VOICE_DECISION_CHECKPOINT"
SETTING_KEY = "voice_decision_checkpoint"

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


def stored_checkpoint() -> Optional[str]:
    from utils.systemone_settings import _owner_setting
    value = _owner_setting(SETTING_KEY)
    return value.strip() or None if isinstance(value, str) else None


def checkpoint_source() -> Optional[str]:
    if stored_checkpoint():
        return "setting"
    return "env" if (os.environ.get(ENV_CHECKPOINT) or "").strip() else None


def checkpoint_name() -> Optional[str]:
    return stored_checkpoint() or (os.environ.get(ENV_CHECKPOINT) or "").strip() or None


def is_decision_checkpoint(name: str) -> bool:
    from core.systemone import catalog
    return hasattr(catalog.resolve(name), "layout")


def set_checkpoint(name: Optional[str]) -> None:
    """Store the pick installation-wide; ``None`` clears it, so the launch env var applies again."""
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    name = (name or "").strip() or None
    if name is not None and not is_decision_checkpoint(name):
        raise ValueError(f"{name} is not a Decision checkpoint")
    run_as(OWNER, upsert_app_settings, {SETTING_KEY: name or ""}, read_back = False)


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


class DecisionUnavailable(RuntimeError):
    """The checkpoint cannot answer right now (loading, or the GPU is taken); routes answer 503."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


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
            raise DecisionUnavailable(
                f"Decision model unavailable: {exc.message}", exc.retry_after
            ) from exc
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
