"""Opt-in embedding shortlist for high-cardinality choice questions.

Choice options share one ``head_max_len`` budget, so a large label set leaves only a few
tokens per label. ``predict_shortlist`` embeds the state and each option with a
caller-supplied ``embed_fn``, keeps the top ``k``, and runs a single ``predict`` (or
``system_one``) on that reduced criteria set.

``Agent.predict`` and ``Agent.system_one`` are separate: they still score every criterion
they are given. This module does not change ``DecisionModel.forward`` and does not add a
second decision-model pass.

The coarse-to-fine pattern is the one the README recommends and the one reported in
https://github.com/NandhaKishorM/laya/issues/102. Ranking here is cosine similarity on
whatever vectors ``embed_fn`` returns. Issue #102's BANKING77 figures belong to that
report; this module does not measure them.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .common import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
) -> List[Any]:
    """Return the top-``k`` choice labels for ``state``.

    ``embed_fn`` maps a list of strings to an array of shape ``(len(texts), dim)``.
    It is called once, with the query text first and then one string per option in
    criteria order. Option strings match ``render_options`` for a choice question.

    When ``k`` is at least the number of labels, every label is returned in its
    original order and ``embed_fn`` is not called.

    Ties keep the earlier label. A zero vector scores 0 and does not outrank a
    label that came before it.
    """
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Shortlist each choice question, then call ``predict`` or ``system_one`` once.

    Non-choice questions are forwarded unchanged. A choice whose label count is
    ``<= k`` is forwarded unchanged and does not call ``embed_fn``. The caller's
    ``questions`` dict is not mutated.

    The returned dict is the model result plus a ``shortlist`` entry. Probabilities
    on a shortlisted choice are over the kept labels only. ``shortlist[qid]`` holds
    ``labels`` (rank order), ``scores`` (cosine, or ``None`` when nothing was
    dropped), ``k``, ``n``, and ``passthrough``.

    Extra keyword arguments are forwarded to ``predict`` / ``system_one`` (for
    example ``model=`` on a ``Router``).
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    checked = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError("question %r is a choice but has no criteria" % (qid,))
        labels, scores, passthrough, n = _rank(
            state, qdef["criteria"], embed_fn, checked, qdef.get("instructions")
        )
        meta[qid] = {
            "labels": list(labels),
            "scores": scores,
            "k": checked,
            "n": n,
            "passthrough": passthrough,
        }
        if passthrough:
            reduced[qid] = qdef
            continue
        updated = dict(qdef)
        updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
        reduced[qid] = updated

    result = _call_predict(agent, state, reduced, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError(
            "predict/system_one must return a dict, got %s" % type(result).__name__
        )
    out = dict(result)
    out["shortlist"] = meta
    return out


def embed_fn_from_agent(
    agent: Any,
    max_length: int = 512,
    batch_size: int = 32,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Mean-pool the checkpoint encoder already loaded on ``agent``.

    The callable embeds a list of strings with ``agent.tok`` and ``agent.model.encoder``.
    It does not run the decision head and does not download weights. A dedicated
    bi-encoder passed as ``embed_fn`` will usually shortlist better; this helper is
    for callers who only have the Laya checkpoint in memory.

    Padding positions are excluded from the mean. The encoder's train/eval flag is
    left as the caller set it (a loaded ``Agent`` is already in eval).
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer, got %r" % (max_length,))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer, got %r" % (batch_size,))

    import torch

    tok = agent.tok
    encoder = agent.model.encoder
    device = agent.device

    def embed_fn(texts: Sequence[str]) -> np.ndarray:
        rows = ["" if text is None else str(text) for text in texts]
        hidden = _hidden_size(encoder)
        if not rows:
            return np.zeros((0, hidden), dtype=np.float32)
        parts: List[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.inference_mode():
                hidden_states = encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            parts.append(pooled.float().cpu().numpy())
        return np.concatenate(parts, axis=0)

    return embed_fn


def _rank(state, criteria, embed_fn, k, instructions):
    checked = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]
    if checked >= n:
        return list(keys), None, True, n
    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = np.argsort(-sims, kind="mergesort")[:checked]
    labels = [keys[int(i)] for i in order]
    scores = [float(sims[int(i)]) for i in order]
    return labels, scores, False, n


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer, got %r" % (k,))
    return k


def _criteria_items(criteria):
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise TypeError(
            "choice criteria must be a dict or list, got %s" % type(criteria).__name__
        )
    if not items:
        raise ValueError("choice criteria must contain at least one option")
    seen = set()
    for key, _value in items:
        if key in seen:
            raise ValueError("choice criteria label %r is duplicated" % (key,))
        seen.add(key)
    return items


def _option_texts(items) -> List[str]:
    crit = {key: value for key, value in items}
    rendered = render_options({"t": "choice", "ins": "", "crit": crit})
    texts = [piece if isinstance(piece, str) else str(piece) for piece in rendered]
    if len(texts) != len(items):
        raise ValueError("could not render every choice option")
    return texts


def _query_text(state, instructions) -> str:
    body = serialize_state(state)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return "%s\n%s" % (instructions, body)


def _subset_criteria(criteria, labels):
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _embeddings(embed_fn, texts: Sequence[str]) -> np.ndarray:
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    raw = embed_fn(list(texts))
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        raise ValueError(
            "embed_fn must return an array of shape (%d, dim), got %s"
            % (len(texts), tuple(arr.shape))
        )
    return np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    qn = float(np.linalg.norm(query))
    dn = np.linalg.norm(docs, axis=1)
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    if qn == 0.0:
        return sims
    denom = dn * qn
    ok = denom > 0.0
    if np.any(ok):
        sims[ok] = docs[ok] @ query / denom[ok]
    return sims


def _call_predict(agent, state, questions, **predict_kwargs):
    fn = getattr(agent, "predict", None)
    if fn is None:
        fn = getattr(agent, "system_one", None)
    if fn is None:
        raise TypeError("agent must provide predict or system_one")
    return fn(state, questions, **predict_kwargs)


def _hidden_size(encoder) -> int:
    size = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return 0
    return size
