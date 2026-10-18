# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The four small voice-mode models: turn judge, filler picker, interject decision, addressed gate.

judge: ModernBERT sequence classifier, P(the user finished the turn), CPU.
filler: int8 ONNX ModernBERT classifier over the ASR text (TF-IDF gate + class pickle when the ONNX
folder is absent), a phrase from phrases.json, an echo slot from the int8 ONNX tagger (rules otherwise).
interject: TF-IDF + hand features over (running prompt, new utterance, elapsed), restart | ignore | queue.
addressed: int8 ONNX ModernBERT over the ASR text plus the previous utterance's label, assistant | other | reading.

Each loads on first use. A model that fails to load (no dir, no torch, bad pickle) stays
unavailable with its reason; nothing here raises past ``available``/``reason``.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Each model is a Hugging Face repo <hub>/voice-<name> with the same layout as the local
# checkout's <name>/ folder. A local dir (UNSLOTH_VOICE_MODELS_DIR, or the checkout) wins when
# it holds the files; otherwise the needed files are fetched into the hub cache.
DEFAULT_DIR = "~/git_dev/unsloth-branch-workshop/voice-endpointing-data"
ENV_DIR = "UNSLOTH_VOICE_MODELS_DIR"
ENV_HUB = "UNSLOTH_VOICE_MODELS_HUB"
DEFAULT_HUB = "Borcherding"
MODELS = ("judge", "filler", "interject", "addressed")

JUDGE_MAX_LENGTH = 64
JUDGE_THREADS = 4
ECHO_MIN_CONF = 0.7
FILLER_ONNX = "modernbert_l4"
JUDGE_ONNX = "modernbert_full"
ECHO_ONNX = "echo_distilbert_l4"
FILLER_MAX_TOKENS = 48
ECHO_MAX_TOKENS = 40
ADDRESSED_ONNX = "modernbert_base"
ADDRESSED_MAX_TOKENS = 64
ADDRESSED_PREVS = ("none", "assistant", "other", "reading")
ORT_THREADS = 4
INTERJECT_ACTIONS = ["restart", "ignore", "queue"]


def models_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR).expanduser()


def models_hub() -> str:
    return (os.environ.get(ENV_HUB) or DEFAULT_HUB).strip()


# What each loader reads, relative to its model dir: the hub download is limited to these.
HUB_FILES = {
    "judge": [f"models/{JUDGE_ONNX}/*", f"models/{JUDGE_ONNX}/onnx/*"],
    "filler": [f"models/{FILLER_ONNX}/*", f"models/{FILLER_ONNX}/onnx*/*", f"models/{ECHO_ONNX}/*", f"models/{ECHO_ONNX}/onnx*/*", "phrases.json"],
    "interject": ["models/tfidf_lr/*"],
    "addressed": [f"models/{ADDRESSED_ONNX}/*", f"models/{ADDRESSED_ONNX}/onnx*/*"],
}
# The one file that says the local folder really holds the model (a stray empty dir does not).
_MARKER = {
    "judge": f"models/{JUDGE_ONNX}/onnx/model_int8.onnx",
    "filler": "phrases.json",
    "interject": "models/tfidf_lr/model.pkl",
    "addressed": f"models/{ADDRESSED_ONNX}/onnx/model_int8.onnx",
}


def resolve_model_dir(name: str, root: Path, hub: Optional[str]) -> Path:
    """The folder holding one model: ``root/<name>`` when it is complete, else the hub snapshot."""
    local = root / name
    if not hub or (local / _MARKER[name]).exists():
        if not root.is_dir():
            raise FileNotFoundError(f"voice models dir not found: {root} (set {ENV_DIR})")
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(f"{hub}/voice-{name}", allow_patterns = HUB_FILES[name]))


# ---------------------------------------------------------------- filler: featurize + echo slot

def featurize(text: str, prev_q: bool = False) -> str:
    return ("<prevq> " if prev_q else "") + text.lower().strip() + " <end>"


_DISFL = {"um", "uh", "like", "you know", "i mean", "basically", "actually", "hmm", "ok", "okay", "please", "sorry", "wait"}
_PREFIX = r"^((hey|hi|hello|yo|so|ok|okay|alright|right|well|now|also|and|quick one|one more thing|please|just|real quick|i was wondering if you could|i was wondering if you can|do you think you could|would you mind|is there any way you could|i need you to|i want you to|i'd like you to|i would like you to|i need help with|i need help|can you|could you|would you|will you|can u|could u|you could|maybe|if you can|if possible|for me)[ ,]+)+"
_STOP = {"that's", "thats", "it's", "its", "who", "whose", "in", "for", "with", "that", "which", "about", "to", "on", "from", "so", "because", "and", "but", "using", "called", "where", "when", "if", "please", "thanks", "real", "quick", "by", "at", "of", "as", "into", "or", "than", "then", "like", "while", "after", "before", "under", "over", "without", "via", "per", "whenever", "once", "every", "until", "unless", "though", "although", "is", "are", "was", "were", "should", "would", "could", "can", "will", "do", "does", "did", "has", "have", "had", "today", "tonight", "tomorrow", "now", "later", "again", "too", "also", "instead", "here", "there", "me", "us", "it", "them", "him", "her", "right", "really", "actually", "basically", "exactly", "maybe", "probably"}
_BAD_SLOT = {"it", "that", "this", "these", "those", "something", "anything", "everything", "one", "some", "more", "stuff", "things", "thing", "you", "me", "us", "them", "the one", "a bit", "a lot", "the same", "the following", "the rest", "the other"}
_DET = r"(?:a|an|the|some|my|your|our|his|her|their|this|that|these|those|one|two|three|four|five|ten|a few|a couple of|a couple|a bunch of|a list of)"
_VERBS_RAW = "any ideas for|ideas for|what should i do about|what should i do with|what can i do about|what's the deal with|whats the deal with|what about|how about|what do you think about|what do you think of|your thoughts on|your take on|how do i deal with|how to deal with|write|writing|make|making|build|building|create|creating|draft|drafting|generate|generating|compose|plan|planning|design|designing|put together|come up with|fix|fixing|debug|debugging|summarize|summarise|rewrite|rephrase|translate|proofread|explain|describe|compare|analyze|analyse|review|check|look at|look into|look up|find|find me|get me|give me|show me|tell me about|tell me|tell|teach me|teach me about|help me with|help me|help with|organize|organise|outline|brainstorm|suggest|recommend|list|name|pick|choose|improve|edit|clean up|optimize|optimise|refactor|set up|install|configure|run|test|calculate|compute|convert|draw|sketch|code|code up|script|implement|add|remove|delete|update|change|turn|schedule|book|prepare|prep|research|dig into|think about|work out|figure out|sort out|do"
_VERBS = "(?:" + "|".join(sorted(_VERBS_RAW.split("|"), key = len, reverse = True)) + ")"
_PROBLEM = r"(?:failing|fails|failed|crashing|crashes|crashed|broken|breaking|erroring|throwing|not working|isn't working|doesn't work|won't work|not loading|won't load|won't start|not starting|won't open|hanging|freezing|frozen|so slow|slow|timing out|dying|not responding|wrong|acting up|messed up|stuck|throws? (?:an )?errors?|throwing (?:an )?errors?|errors out|gives (?:me )?(?:an )?error|keeps? erroring|returns? (?:an )?error|(?:is|keeps) (?:giving|showing) (?:me )?(?:an )?error)"
_LIGHT_PREP = {"to", "for", "on", "about", "of"}
_FLIP = {"my": "your", "our": "your", "mine": "yours", "i": "you", "me": "you", "myself": "yourself", "i'm": "you're", "i've": "you've", "we": "you", "us": "you"}


def _clean(t: str) -> str:
    t = re.sub(r"(\w)-(\w)", r"\1\2", t.lower())
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\b(you know|i mean|kind of|sort of)\b", " ", t)
    out: list[str] = []
    for x in (w for w in t.split() if w not in _DISFL):
        if out and out[-1] == x:
            continue
        out.append(x)
    return " ".join(out)


def _finish(span: Optional[str], problem: bool = False) -> Optional[str]:
    if not span:
        return None
    w = [_FLIP.get(x, x) for x in span.strip().split()]
    while w and w[-1] in _STOP:
        w.pop()
    while w and w[0] in _STOP - {"the", "a", "an"}:
        w.pop(0)
    if not w:
        return None
    if problem:
        w = [x for x in w if x not in ("the", "a", "an", "this", "that", "your", "my", "our")]
        if not w:
            return None
        if w[-1] not in ("error", "issue", "problem", "bug", "crash"):
            w = w + ["error"]
        w = ["the"] + w
    w = w[:5]
    s = " ".join(w)
    if s in _BAD_SLOT or all(x in _STOP or x in {"a", "an", "the", "your"} for x in w):
        return None
    if len(w) == 1 and w[0] in {"your", "the", "a", "an", "some"}:
        return None
    if len(w) == 1:
        return None if w[0] in ("you", "yours") else "the " + s
    return s


def _np(words: list[str], maxw: int = 5) -> str:
    out: list[str] = []
    for x in words:
        if x in _STOP or len(out) >= maxw:
            break
        out.append(x)
    if out and len(out) < len(words) and words[len(out)] in _LIGHT_PREP:
        tail: list[str] = []
        for x in words[len(out) + 1 :]:
            if x in _STOP or x in _LIGHT_PREP:
                break
            tail.append(x)
        while tail and tail[-1] in {"my", "your", "a", "an", "the", "some", "this", "that"}:
            tail.pop()
        if tail and len(out) + 1 + len(tail) <= maxw:
            out += [words[len(out)]] + tail
    return " ".join(out)


def extract_slot(text: str) -> tuple[Optional[str], float]:
    """A 2-5 word noun phrase naming what was asked about (pronouns flipped), with a confidence."""
    D, V, P = _DET, _VERBS, _PROBLEM
    t = re.sub(_PREFIX, "", _clean(text)).strip()
    if not t or len(t.split()) <= 2:
        return None, 0.0

    def take(det, rest, maxw = 5, conf_det = 0.9, conf_nodet = 0.7, problem = False):
        core = _np(rest.split(), maxw)
        if not core:
            return None
        s = _finish(((det + " ") if det else "") + core, problem = problem)
        return (s, conf_det if det else conf_nodet) if s else None

    m = re.match(rf"^(?:why|how come)(?: is| does| did| are| won't| isn't| doesn't| do| has| keeps| keep)? (?P<det>{D})? ?(?P<np>(?:\w+ ){{1,3}}?)(?:keeps? |still |always |suddenly |now |just )*{P}", t)
    if m:
        s = _finish(m.group("np"), problem = True)
        if s:
            return s, 0.85
    m = re.match(rf"^(?P<det>{D}) (?P<np>(?:\w+ ){{1,3}}?)(?=is |are |keeps |keep |won't |doesn't |has |just |suddenly |still |isn't |wont |doesnt |crash|fail|throw|break|broke).*?{P}", t)
    if m:
        s = _finish(m.group("np"), problem = True)
        if s:
            return s, 0.7
    m = re.match(rf"^(?:{V}) (?:to me |for me |me |us |up |out |to )?(?:(?P<det>{D}) )?(?P<rest>.+)$", t)
    if m:
        if not m.group("det") and re.match(rf"^(?:{V}) ", m.group("rest")):
            return extract_slot(m.group("rest"))
        r = take(m.group("det"), m.group("rest"))
        if r:
            return r
    m = re.match(rf"^how (?:do|can|should|would|could|to) (?:i|you|we|one) (?:{V}|\w+) (?:(?P<det>{D}) )?(?P<rest>.+)$", t)
    if m:
        r = take(m.group("det"), m.group("rest"), conf_det = 0.7, conf_nodet = 0.6)
        if r:
            return r
    m = re.match(rf"^(?:what|what's|whats|who|who's) (?:is |are |was |were |exactly is |does |do )?(?:(?P<det>{D}) )?(?P<rest>.+)$", t)
    if m:
        r = take(m.group("det"), m.group("rest"), maxw = 4, conf_det = 0.55, conf_nodet = 0.5)
        if r and len(r[0].split()) >= 2:
            return r
    m = re.match(rf"^how (?:does|do|did) (?:(?P<det>{D}) )?(?P<rest>.+?) (?:work|works|function|happen|happens)\b", t)
    if m:
        r = take(m.group("det"), m.group("rest"), conf_det = 0.7, conf_nodet = 0.6)
        if r:
            return r
    m = re.match(r"^(?:my|our) (?P<np>(?:\w+ ){1,3}?)(?:is|are|keeps|keep|won't|doesn't|has|have|just|was|were) ", t)
    if m:
        s = _finish("your " + m.group("np"))
        if s:
            return s, 0.6
    m = re.match(rf"^(?:i need help with|i need a hand with|i need|i want|i'd like|i would like|i'm looking for|im looking for|looking for|i have a question about|question about|i'm trying to (?:\w+ ){{0,2}}) ?(?:you to (?:\w+ ))?(?:(?P<det>{D}) )?(?P<rest>.+)$", t)
    if m:
        r = take(m.group("det"), m.group("rest"), conf_det = 0.6, conf_nodet = 0.5)
        if r:
            return r
    return None, 0.0


def _ort_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    return ort.InferenceSession(str(path), so, providers = ["CPUExecutionProvider"])


def _softmax(x):
    import numpy as np

    e = np.exp(x - x.max(-1, keepdims = True))
    return e / e.sum(-1, keepdims = True)


class _OnnxClassifier:
    def __init__(self, d: Path):
        from tokenizers import Tokenizer

        self.classes = json.load((d / "classes.json").open())
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(FILLER_MAX_TOKENS)
        self.session = _ort_session(d / "model_int8.onnx")
        self.inputs = {x.name for x in self.session.get_inputs()}

    def classify(self, text: str, prev_q: bool) -> tuple[str, float]:
        import numpy as np

        e = self.tok.encode(featurize(text, prev_q))
        feed = {"input_ids": np.array([e.ids], dtype = np.int64), "attention_mask": np.array([e.attention_mask], dtype = np.int64)}
        p = _softmax(self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0])
        i = int(p.argmax())
        return self.classes[i], float(p[i])


class _TfidfClassifier:
    def __init__(self, d: Path):
        m = pickle.load((d / "model.pkl").open("rb"))
        self.gate, self.cls = m["gate"], m["cls"]

    def classify(self, text: str, prev_q: bool) -> tuple[str, float]:
        x = featurize(text, prev_q)
        p_say = float(self.gate.predict_proba([x])[0, 1])
        if p_say < 0.5:
            return "none", 1 - p_say
        probs = self.cls.predict_proba([x])[0]
        i = int(probs.argmax())
        return str(self.cls.classes_[i]), float(probs[i])


class _OnnxEchoTagger:
    """Per-token B/I/O logits; longest span wins, then the same finish() as the rules."""

    def __init__(self, d: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(ECHO_MAX_TOKENS)
        self.session = _ort_session(d / "model_int8.onnx")

    def extract(self, text: str) -> tuple[Optional[str], float]:
        import numpy as np

        e = self.tok.encode(text)
        logits = self.session.run(None, {"input_ids": np.array([e.ids], dtype = np.int64), "attention_mask": np.array([e.attention_mask], dtype = np.int64)})[0][0]
        p = _softmax(logits)
        pred = p.argmax(-1)
        spans: list[tuple[list[int], list[float]]] = []
        cur: Optional[list[int]] = None
        confs: list[float] = []
        for j, (a, b) in enumerate(e.offsets):
            if a == b:
                continue
            if pred[j] == 1 or (pred[j] == 2 and cur is None):
                if cur:
                    spans.append((cur, confs))
                cur, confs = [a, b], [float(p[j].max())]
            elif pred[j] == 2 and cur is not None:
                cur[1] = b
                confs.append(float(p[j].max()))
            elif cur:
                spans.append((cur, confs))
                cur, confs = None, []
        if cur:
            spans.append((cur, confs))
        if not spans:
            return None, 0.0
        (a, b), confs = max(spans, key = lambda s: s[0][1] - s[0][0])
        problem = bool(re.search(_PROBLEM, text[b:])) or bool(re.match(r"^(why|how come)", text.strip()))
        s = _finish(text[a:b], problem = problem)
        return (s, round(sum(confs) / len(confs), 3)) if s else (None, 0.0)


class _RulesEchoTagger:
    extract = staticmethod(extract_slot)


class _Filler:
    def __init__(self, root: Path, seed: Optional[int] = None):
        models = root / "models"
        if (models / FILLER_ONNX).is_dir():
            self.clf, self.backend = _OnnxClassifier(models / FILLER_ONNX), "onnx"
        else:
            self.clf, self.backend = _TfidfClassifier(models / "tfidf_lr"), "tfidf"
        if (models / ECHO_ONNX).is_dir():
            self.echo, self.echo_backend = _OnnxEchoTagger(models / ECHO_ONNX), "onnx"
        else:
            self.echo, self.echo_backend = _RulesEchoTagger(), "rules"
        self.bank = json.load((root / "phrases.json").open())
        self.rng = random.Random(seed)

    def _pick(self, items: list[dict]) -> str:
        return self.rng.choices([p["text"] for p in items], weights = [p["weight"] for p in items])[0]

    def predict(self, text: str, prev_q: bool = False) -> dict:
        t0 = time.perf_counter()
        cls, conf = self.clf.classify(text, prev_q)
        out: dict[str, Any] = {"say": cls != "none", "class": cls, "confidence": round(conf, 3), "phrase": "", "slot": None, "echo": False}
        if cls != "none":
            out["phrase"] = self._pick(self.bank["classes"][cls]["phrases"])
            if cls != "minimal":  # a one-word tier stays one word
                slot, sconf = self.echo.extract(text)
                out["slot"] = slot
                echo = self.bank["echo_templates"]
                if slot and sconf >= ECHO_MIN_CONF and self.rng.random() < echo["mix"]["echo_when_slot"]:
                    s = self._pick(echo["templates"]).replace("{slot}", slot)
                    out["phrase"] = s[0].upper() + s[1:]
                    out["echo"] = True
        out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out


# ---------------------------------------------------------------- interject: hand features + bias

_IJ_STOP = set("a an the to of in on for and or but so um uh like you know i me my we our it its this that these those is are was were be been do does did can could would should will please just really very some any with at by from as about what which who how why when where into onto up down out over if then than there here yeah yes no ok okay oh hmm well right".split())
_RESTART_CUE = r"^(um |uh |so |oh )?(no|wait|actually|instead|scratch|hold on|sorry|i meant|make it|and |also |plus |oh and|with |can it|could it|could you|can you|what about|how about|is there a way)\b"
_QUEUE_CUE = r"\b(remind|timer|alarm|weather|unrelated|separate|different question|another thing|one more thing|when you're done|after (that|this)|what time|turn (the|it)|play |order |text |call |email |calendar|meeting)\b"
_IGNORE_CUE = r"^(um |uh |so |oh )?(ok|okay|mm|uh huh|yeah|yep|right|sure|alright|cool|hmm|take your time|no rush|go ahead|go on|sorry go|one sec|hang on|i see|got it|nice|great|perfect|awesome|never mind|where|ugh|oh|ah|ha|honey|sweetie|dude|mom|dad|guys|hey|coming|yeah i)\b"
_CUE_WORDS = ["no", "wait", "actually", "instead", "and", "also", "plus", "oh", "can", "could", "what", "how", "ok", "okay", "yeah", "hmm", "sorry", "remind", "set", "after", "when", "then", "um", "uh", "the", "take", "where", "not", "make", "in", "with", "for", "add", "it"]
_CORRECTION_CUE = re.compile(r"^(um |uh |so |oh |hmm )*(no|wait|actually|instead|scratch that|hold on|sorry|i meant|not that|wrong)\b")
_ADD_LEAD = re.compile(r"^(um |uh |so |oh )*(and also|oh and|and|also|plus|with|oh also)\s+")


def _content(t: str) -> set[str]:
    return {x for x in re.sub(r"[^\w\s']", " ", t.lower()).split() if x not in _IJ_STOP and len(x) > 2}


def hand_features(prompt: str, utt: str, elapsed: float) -> list[float]:
    """Fixed order; the pickled scaler was fit on exactly this layout."""
    cp, cu = _content(prompt), _content(utt)
    uw = utt.split()
    first = uw[0] if uw else ""
    first2 = " ".join(uw[:2])
    inter = len(cp & cu)
    f = [
        inter / max(1, len(cu)),
        inter / max(1, len(cp | cu)),
        float(inter > 0), float(inter >= 2),
        min(len(uw), 30) / 30, min(len(cu), 15) / 15, float(len(uw) <= 2), float(len(uw) <= 4),
        min(elapsed, 30) / 30, float(elapsed < 1.5), float(elapsed < 4), float(elapsed > 10), math.log1p(elapsed) / 4,
        float(bool(re.search(_RESTART_CUE, utt))), float(bool(re.search(_QUEUE_CUE, utt))), float(bool(re.search(_IGNORE_CUE, utt))),
        float(utt.rstrip().endswith("?")), float(first in ("no", "wait", "actually", "instead")),
        float(first2 in ("oh and", "and also", "and add", "and make", "and keep")),
        float(bool(re.search(r"\b(it|that|this|them|the (first|second|last|other) one)\b", utt))),
        float(bool(re.search(r"\b(not you|honey|sweetie|dude|guys|be right there|coming|one sec|on a call)\b", utt))),
    ]
    f += [float(first == w) for w in _CUE_WORDS]
    return f


def merge_prompt(prompt: str, utterance: str) -> str:
    u = utterance.strip()
    if _CORRECTION_CUE.search(u):
        return f"{prompt.rstrip('. ')}. Correction: {u}"
    return f"{prompt.rstrip('. ')}. Also: {_ADD_LEAD.sub('', u)}"


def apply_runtime_bias(probs: dict[str, float], utterance: str, elapsed_s: float) -> dict[str, float]:
    pr = dict(probs)
    if elapsed_s < 1.5:
        pr["ignore"] *= 1.5
    u = utterance.strip()
    bare = re.fullmatch(r"((um|uh|no|wait|actually|instead|hmm|oh)\s*)+", u)
    other = re.search(r"\b(not you|not now|you're fine|it's fine|that's fine)\b", u)
    if re.match(r"^(um |uh )?(no|wait|actually|instead)\b", u) and not bare and not other:
        pr["restart"] *= 1.5
    if other:
        pr["ignore"] *= 4.0  # said to someone else, never a reprompt
    z = sum(pr.values())
    return {k: v / z for k, v in pr.items()}


class _InterjectTfidf:
    """Shape of the pickled ``train_tfidf.Model``: attributes are sklearn objects, methods live here."""

    def _x(self, rows: list[dict]):
        import numpy as np
        import scipy.sparse as sp

        u = [r["utterance"] for r in rows]
        p = [r["prompt"] for r in rows]
        h = np.array([hand_features(r["prompt"], r["utterance"], r["elapsed_s"]) for r in rows])
        parts = [self.vu_w.transform(u), self.vu_c.transform(u), 0.4 * self.vp_w.transform(p), sp.csr_matrix(self.sc.transform(h))]
        return sp.hstack(parts).tocsr()

    def proba(self, rows: list[dict]):
        return self.clf.predict_proba(self._x(rows))


class _InterjectUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) == ("train_tfidf", "Model"):
            return _InterjectTfidf
        return super().find_class(module, name)


class _Interject:
    backend = "tfidf"

    def __init__(self, root: Path):
        with (root / "models" / "tfidf_lr" / "model.pkl").open("rb") as f:
            self.m = _InterjectUnpickler(f).load()

    def proba(self, prompt: str, utterance: str, elapsed_s: float) -> dict[str, float]:
        P = self.m.proba([{"prompt": prompt, "utterance": utterance, "elapsed_s": elapsed_s}])[0]
        return {str(c): float(p) for c, p in zip(self.m.clf.classes_, P)}

    def predict(self, prompt: str, utterance: str, elapsed_s: float = 2.0) -> dict:
        pr = apply_runtime_bias(self.proba(prompt, utterance, elapsed_s), utterance, elapsed_s)
        action = max(pr, key = pr.get)
        return {
            "action": action,
            "confidence": round(pr[action], 4),
            "probs": {k: round(v, 4) for k, v in pr.items()},
            "merged_prompt": merge_prompt(prompt, utterance) if action == "restart" else None,
        }


# ---------------------------------------------------------------- judge

class _OnnxJudge:
    backend = "onnx"

    def __init__(self, d: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(JUDGE_MAX_LENGTH)
        self.session = _ort_session(d / "onnx" / "model_int8.onnx")
        self.inputs = {x.name for x in self.session.get_inputs()}
        self("warm up")

    def __call__(self, text: str) -> float:
        import numpy as np

        e = self.tok.encode(text)
        feed = {"input_ids": np.array([e.ids], dtype = np.int64), "attention_mask": np.array([e.attention_mask], dtype = np.int64)}
        p = _softmax(self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0])
        return float(p[1])


class _TorchJudge:
    backend = "modernbert"

    def __init__(self, d: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(JUDGE_THREADS)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d, attn_implementation = "sdpa").eval()
        self("warm up")

    def __call__(self, text: str) -> float:
        with self.torch.no_grad():
            enc = self.tok([text], truncation = True, max_length = JUDGE_MAX_LENGTH, return_tensors = "pt")
            return self.torch.softmax(self.model(**enc).logits.float(), -1)[0, 1].item()


def _Judge(root: Path):
    # The full-data int8 export when present (4 ms), else the first-pass torch weights.
    onnx = root / "models" / JUDGE_ONNX
    if (onnx / "onnx" / "model_int8.onnx").exists():
        return _OnnxJudge(onnx)
    return _TorchJudge(root / "models" / "modernbert_base")


# ---------------------------------------------------------------- addressed

def featurize_addressed(text: str, prev: str = "none") -> str:
    """The context token names what came before: the assistant's own reply counts as ``assistant``."""
    if prev not in ADDRESSED_PREVS:
        prev = "none"
    return f"<prev:{prev}> {text.lower().strip()} <end>"


class _OnnxAddressed:
    backend = "onnx"

    def __init__(self, d: Path):
        from tokenizers import Tokenizer

        self.classes = json.load((d / "classes.json").open())
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(ADDRESSED_MAX_TOKENS)
        self.session = _ort_session(d / "onnx" / "model_int8.onnx")
        self.inputs = {x.name for x in self.session.get_inputs()}
        self.predict("warm up")

    def predict(self, text: str, prev: str = "none") -> dict:
        import numpy as np

        e = self.tok.encode(featurize_addressed(text, prev))
        feed = {"input_ids": np.array([e.ids], dtype = np.int64), "attention_mask": np.array([e.attention_mask], dtype = np.int64)}
        p = _softmax(self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0])
        i = int(p.argmax())
        return {
            "label": self.classes[i],
            "confidence": round(float(p[i]), 4),
            "probs": {c: round(float(x), 4) for c, x in zip(self.classes, p)},
        }


def _Addressed(root: Path):
    return _OnnxAddressed(root / "models" / ADDRESSED_ONNX)


# ---------------------------------------------------------------- singleton

class VoiceModels:
    def __init__(self, root: Optional[Path] = None, hub: Optional[str] = None):
        # An explicit root is local-only (tests, a pinned checkout); the default also fetches from the hub.
        self.root = Path(root) if root else models_dir()
        self.hub = hub if root else (hub or models_hub())
        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._dirs: dict[str, Path] = {}

    _loaders = {"judge": _Judge, "filler": _Filler, "interject": _Interject, "addressed": _Addressed}

    def _get(self, name: str) -> Any:
        if name in self._models:
            return self._models[name]
        with self._lock:
            if name in self._models:
                return self._models[name]
            if name in self._errors:
                return None
            try:
                self._dirs[name] = resolve_model_dir(name, self.root, self.hub)
                self._models[name] = self._loaders[name](self._dirs[name])
            except Exception as exc:  # noqa: BLE001 - the model stays unavailable, the route answers 503
                self._errors[name] = f"{type(exc).__name__}: {exc}"
                return None
            return self._models[name]

    def available(self, name: str) -> bool:
        return self._get(name) is not None

    def reason(self, name: str) -> str:
        self._get(name)
        return self._errors.get(name, "")

    def backend(self, name: str) -> Optional[str]:
        m = self._get(name)
        return getattr(m, "backend", None) if m is not None else None

    def status(self) -> dict:
        out: dict[str, Any] = {name: self.available(name) for name in MODELS}
        out["dir"] = str(self.root)
        out["hub"] = self.hub
        out["dirs"] = {name: str(path) for name, path in self._dirs.items()}
        out["backends"] = {name: self.backend(name) for name in MODELS}
        filler = self._models.get("filler")
        out["backends"]["filler_echo"] = getattr(filler, "echo_backend", None)
        out["errors"] = dict(self._errors)
        return out

    def judge(self, text: str) -> float:
        return self._get("judge")(text)

    def filler(self, text: str, prev_q: bool = False) -> dict:
        return self._get("filler").predict(text, prev_q)

    def interject(self, prompt: str, utterance: str, elapsed_s: float = 2.0) -> dict:
        return self._get("interject").predict(prompt, utterance, elapsed_s)

    def addressed(self, text: str, prev: str = "none") -> dict:
        return self._get("addressed").predict(text, prev)


_instance: Optional[VoiceModels] = None
_instance_lock = threading.Lock()


def get_voice_models() -> VoiceModels:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = VoiceModels()
        return _instance
