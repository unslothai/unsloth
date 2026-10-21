# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Units are frozen and all distinct, since Shiki caches output keyed on the source string."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Optional

FROZEN_DIR = Path(__file__).resolve().parent / "corpus" / "frozen"
UNITS_JSONL = FROZEN_DIR / "units.jsonl"
MANIFEST_JSON = FROZEN_DIR / "manifest.json"

CORPUS_SEED = 20260819
# Bump whenever generated text changes: floor_table refuses to pool numbers across versions.
CORPUS_VERSION = 2

# Calibrated against the field capture; these are means of a jittered distribution.
PROSE_CHARS = 1250
FENCE_CHARS = 1800
PREAMBLE_FRACTION = 0.25

# Jitter decorrelates per-block from per-char cost; uses the per-unit rng so units stay stable.
BLOCK_JITTER = 0.55

UNIT_JITTER = 0.35

TOOL_CALLS_PER_UNIT = (0, 3)

#: The stored part shape that actually renders a tool block. VERIFIED, not assumed: a flat
#: `{"type": "tool-call"}` and an assistant-ui `{"type": "tool-invocation"}` both rendered
#: their sibling text and NO tool UI, while this shape produced a "Used tool" group.
TOOL_NAMES = ("web_search", "code_execution", "python", "terminal", "search_knowledge_base")

CYCLE_BASE = ((("reasoning", 10_000), ("code", 8_000)),)
# Cap doubling, else the 1M rung is a single 2.5M-char turn.
MAX_UNIT_CHARS = 320_000

# Units past this many chars are regenerated locally and checked against the manifest hash.
SHIPPED_CHARS_BUDGET = 460_000


_NOUNS = (
    "buffer",
    "scheduler",
    "fibre",
    "observer",
    "commit",
    "layout",
    "token",
    "shard",
    "lane",
    "chunk",
    "boundary",
    "descriptor",
    "checkpoint",
    "gradient",
    "kernel",
    "adapter",
    "window",
    "residual",
    "quantiser",
    "allocator",
    "transcript",
    "viewport",
    "sentinel",
    "digest",
)
_VERBS = (
    "reparses",
    "flushes",
    "clones",
    "invalidates",
    "schedules",
    "accumulates",
    "walks",
    "coalesces",
    "materialises",
    "spills",
    "reconciles",
    "hoists",
    "pins",
    "drains",
)
_ADJS = (
    "cumulative",
    "monotonic",
    "deferred",
    "synchronous",
    "transient",
    "bounded",
    "retained",
    "speculative",
    "interleaved",
    "hysteretic",
    "collinear",
    "saturating",
)
_CONNECTIVES = (
    "which means",
    "so in practice",
    "and therefore",
    "but only when",
    "except that",
    "as long as",
    "provided that",
    "which is why",
)
_LANGS = ("python", "typescript", "rust", "go", "c")
# Span density is per token, so long names dilute it.
_SHORT = (
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "n",
    "p",
    "q",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "acc",
    "buf",
    "idx",
    "key",
    "len",
    "map",
    "ptr",
    "res",
    "tmp",
)

# Both delimiter families: remark-math consumes $...$, preprocessLaTeX rewrites \(...\).
_MATH_OPS = ("+", "-", "\\cdot", "\\times", "\\oplus")
_MATH_RELS = ("=", "\\le", "\\ge", "\\approx", "\\equiv")
_MATH_FUNCS = ("\\log", "\\exp", "\\sin", "\\cos", "\\tanh")
_MATH_GREEK = (
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\delta",
    "\\theta",
    "\\lambda",
    "\\mu",
    "\\sigma",
    "\\phi",
    "\\omega",
)

# Math replaces a prose block with the same size distribution, so fence calibration is unchanged.
MATH_BLOCK_PROB = 0.16

# Inline math is spent from the prose block's own budget.
INLINE_MATH_PROB = 0.22

# The preamble stays pure prose so the onset of span cost is visible against it.


@dataclass(frozen = True)
class Unit:
    """One assistant turn's worth of content."""

    index: int
    kind: str
    reasoning: str
    content: str
    chars: int
    sha256: str
    # Not counted in `chars`, which must stay comparable with earlier measurements.
    tool_calls: tuple = ()

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def as_row(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "chars": self.chars,
            "sha256": self.sha256,
            "reasoning": self.reasoning,
            "content": self.content,
            "tool_calls": list(self.tool_calls),
        }

    def clipped_to(self, chars: int) -> "Unit":
        """Clips on block boundaries so no fence is left unclosed, which would take a different
        render path."""
        if chars >= self.chars:
            return self

        def take(text: str, budget: int) -> str:
            blocks = text.split("\n\n")
            out: list[str] = []
            size = 0
            for b in blocks:
                if out and size + len(b) + 2 > budget:
                    break
                out.append(b)
                size += len(b) + 2
            return "\n\n".join(out)

        reasoning_budget = max(1, int(chars * (len(self.reasoning) / self.chars)))
        reasoning = take(self.reasoning, reasoning_budget)
        content = take(self.content, max(1, chars - len(reasoning)))
        text = reasoning + content
        return Unit(
            index = self.index,
            kind = self.kind,
            reasoning = reasoning,
            content = content,
            chars = len(text),
            # Different digest: a clipped unit must not be checked against the frozen unit's hash.
            sha256 = "clip:"
            + hashlib.sha256(f"{self.sha256}\x00{chars}".encode("utf-8")).hexdigest(),
            # Keep tool calls when clipping; they carry no chars and the smallest rung is the ratio base.
            tool_calls = self.tool_calls,
        )

    @classmethod
    def from_row(cls, row: dict) -> "Unit":
        return cls(
            index = row["index"],
            kind = row["kind"],
            reasoning = row["reasoning"],
            content = row["content"],
            chars = row["chars"],
            sha256 = row["sha256"],
            tool_calls = tuple(row.get("tool_calls") or ()),
        )


def _expression(rng: random.Random, salt: str, terms: int) -> str:
    """Salt sits in a subscript, not a comment, so no normaliser can strip it and restore the cache hit."""
    parts = [f"{rng.choice(_MATH_GREEK)}_{{{salt}}}", rng.choice(_MATH_RELS)]
    for i in range(terms):
        if i:
            parts.append(rng.choice(_MATH_OPS))
        shape = rng.random()
        if shape < 0.30:
            parts.append(
                f"\\frac{{{rng.choice(_MATH_GREEK)}^{{{rng.randint(2, 9)}}}}}"
                f"{{{rng.randint(1, 99)} {rng.choice(_MATH_GREEK)}}}"
            )
        elif shape < 0.50:
            parts.append(
                f"\\sum_{{{rng.choice(_SHORT)}={rng.randint(0, 3)}}}^{{{rng.randint(4, 99)}}} "
                f"{rng.choice(_MATH_GREEK)}_{{{rng.choice(_SHORT)}}}"
            )
        elif shape < 0.65:
            parts.append(f"\\sqrt{{{rng.choice(_MATH_GREEK)} {rng.randint(2, 99)}}}")
        elif shape < 0.80:
            parts.append(
                f"{rng.choice(_MATH_FUNCS)}\\left({rng.choice(_MATH_GREEK)}"
                f"^{{{rng.randint(2, 5)}}}\\right)"
            )
        else:
            parts.append(f"{rng.choice(_MATH_GREEK)}^{{{rng.randint(2, 9)}}}")
    return " ".join(parts)


def _inline_math(rng: random.Random, salt: str) -> str:
    """The backslash-paren form only reaches the renderer through preprocessLaTeX, not markdown syntax."""
    body = _expression(rng, salt, rng.randint(1, 2))
    return f"$ {body} $" if rng.random() < 0.65 else f"\\( {body} \\)"


def _sentence(
    rng: random.Random,
    salt: str,
    *,
    math: bool = False,
) -> str:
    tail = f" where {_inline_math(rng, salt)} holds" if math else ""
    return (
        f"The {rng.choice(_ADJS)} {rng.choice(_NOUNS)} {rng.choice(_VERBS)} the "
        f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}_{salt}, {rng.choice(_CONNECTIVES)} the "
        f"{rng.choice(_NOUNS)} stays {rng.choice(_ADJS)}{tail}."
    )


def _prose(
    rng: random.Random,
    target: int,
    salt: str,
    *,
    math: bool = False,
) -> str:
    """Fence-free prose; math is spent from target so block sizes keep their calibrated distribution."""
    out: list[str] = []
    size = 0
    while size < target:
        para: list[str] = []
        for _ in range(rng.randint(3, 6)):
            s = _sentence(rng, salt, math = math and rng.random() < INLINE_MATH_PROB)
            para.append(s)
            size += len(s) + 1
        out.append(" ".join(para))
        size += 2
        if size >= target:
            break
    return "\n\n".join(out)


def _math_block(rng: random.Random, target: int, salt: str) -> str:
    """Many display blocks, not one big equation, so the cost matches what accumulates in long threads."""
    out: list[str] = []
    size = 0
    i = 0
    while size < target:
        body = _expression(rng, f"{salt}m{i}", rng.randint(3, 6))
        block = f"$$\n{body}\n$$" if rng.random() < 0.65 else f"\\[\n{body}\n\\]"
        out.append(block)
        size += len(block) + 2
        i += 1
        if size >= target:
            break
        gloss = _prose(rng, _jitter(rng, 260, floor = 60), f"{salt}g{i}", math = True)
        out.append(gloss)
        size += len(gloss) + 2
    return "\n\n".join(out)


def _fence(
    rng: random.Random,
    target: int,
    salt: str,
    lang: Optional[str] = None,
) -> str:
    """Salt goes in identifiers, not comments, since a highlighter could normalise comments away."""
    lang = lang or rng.choice(_LANGS)
    lines = [f"```{lang}"]
    size = len(lines[0]) + 1
    i = 0
    # Short names: the density target is spans, not characters (field capture: 5.6 chars/span).
    r = rng.randint

    def v() -> str:
        # Mix short and long names to approximate the field's 5.6 chars/span.
        return rng.choice(_SHORT) if rng.random() < 0.55 else rng.choice(_NOUNS)

    while size < target:
        # One salted identifier per line suffices to defeat Shiki's source-keyed cache.
        u = f"{rng.choice(_SHORT)}{salt}{i}"
        if lang == "python":
            line = (
                f"    {u}={v()}[{r(0,9)}]*{r(2,99)}-{v()}({v()},{v()})+{v()}[{r(0,9)}:{r(1,9)}]"
                f";{v()}={v()}%{r(2,9)} if {v()}>{r(1,99)} else {v()}|{r(1,7)}"
            )
        elif lang == "typescript":
            line = (
                f"  const {u}={{a:{v()}[{r(0,9)}],b:{v()}({v()},{r(1,99)}),c:{v()}?{v()}:"
                f"{r(0,9)}}};{v()}={v()}&{r(1,7)}|{v()}<<{r(1,7)};"
            )
        elif lang == "rust":
            line = (
                f"    let {u}:u64={v()}[{r(0,9)}]^{v()}({v()},{r(1,99)})&{r(1,255)};"
                f"{v()}={v()}.iter().map(|x|x*{r(2,99)}).sum::<u64>();"
            )
        elif lang == "go":
            line = (
                f"\t{u}:={v()}[{r(0,9)}]+{v()}({v()},{r(1,99)})*{r(2,99)};"
                f"{v()},{v()}={v()}%{r(2,9)},{v()}>>{r(1,7)}"
            )
        else:
            line = (
                f"    uint64_t {u}={v()}[{r(0,9)}]|({v()}({v()},{r(1,99)})&0x{r(16,255):02x})"
                f";{v()}=*{v()}++^{r(1,255)};"
            )
        lines.append(line)
        size += len(line) + 1
        i += 1
    lines.append("```")
    return "\n".join(lines)


def _jitter(
    rng: random.Random,
    mean: int,
    spread: float = BLOCK_JITTER,
    floor: int = 80,
) -> int:
    """Uniform spread rather than lognormal, so the mean stays where the span-density calibration put it."""
    return max(floor, int(mean * (1.0 + rng.uniform(-spread, spread))))


def _body(rng: random.Random, target: int, salt: str, *, preamble: bool) -> str:
    """Preamble, then a jittered prose/fence alternation, to `target` characters."""
    parts: list[str] = []
    size = 0
    if preamble:
        want = int(target * PREAMBLE_FRACTION)
        head = _prose(rng, want, f"{salt}p")
        parts.append(head)
        size += len(head) + 2
    while size < target:
        want = _jitter(rng, PROSE_CHARS)
        if rng.random() < MATH_BLOCK_PROB:
            p = _math_block(rng, want, f"{salt}{len(parts)}")
        else:
            p = _prose(rng, want, f"{salt}{len(parts)}", math = True)
        parts.append(p)
        size += len(p) + 2
        if size >= target:
            break
        f = _fence(rng, _jitter(rng, FENCE_CHARS), f"{salt}{len(parts)}")
        parts.append(f)
        size += len(f) + 2
    return "\n\n".join(parts)


def _tool_calls(rng: random.Random, salt: str) -> list[dict]:
    """Zero to three tool calls for one turn, in the part shape the app actually renders."""
    low, high = TOOL_CALLS_PER_UNIT
    out: list[dict] = []
    for i in range(rng.randint(low, high)):
        name = rng.choice(TOOL_NAMES)
        query = _prose(rng, _jitter(rng, 220, floor = 40), f"{salt}t{i}q").replace("\n", " ")
        result = _prose(rng, _jitter(rng, 900, floor = 120), f"{salt}t{i}r")
        args = (
            {"query": query} if name in ("web_search", "search_knowledge_base") else {"code": query}
        )
        out.append(
            {
                "type": "tool-call",
                "toolCallId": f"call_{salt}_{i}",
                "toolName": name,
                "argsText": json.dumps(args),
                "args": args,
                "state": "result",
                "result": result,
            }
        )
    return out


def _unit_targets(index: int) -> tuple[str, int]:
    """(kind, target chars) for unit `index` of the escalating cycle."""
    cycle = index // 2
    slot = index % 2
    base = 10_000 if slot == 0 else 8_000
    kind = "reasoning" if slot == 0 else "code"
    nominal = min(MAX_UNIT_CHARS, base * (2**cycle))
    # Separate RNG seeded by index alone, so changing block jitter does not reshuffle unit sizes.
    rng = random.Random((index * 6_364_136_223_846_793_005) ^ 0x5DEECE66D)
    return kind, max(1_500, min(MAX_UNIT_CHARS, _jitter(rng, nominal, UNIT_JITTER, floor = 1_500)))


def generate_unit(index: int, seed: int = CORPUS_SEED) -> Unit:
    """Seeded per unit, not per corpus, so every rung shares the same prefix of units."""
    rng = random.Random((seed * 1_000_003) ^ (index * 2_654_435_761))
    kind, target = _unit_targets(index)
    salt = f"{index:04d}"
    if kind == "reasoning":
        reasoning = _body(rng, int(target * 0.8), f"r{salt}", preamble = True)
        content = _body(rng, int(target * 0.2), f"a{salt}", preamble = False)
    else:
        reasoning = _prose(rng, 1_200, f"r{salt}")
        content = _body(rng, target, f"a{salt}", preamble = False)
    tools = tuple(_tool_calls(rng, salt))
    chars = len(reasoning) + len(content)
    digest = hashlib.sha256(
        (
            f"{index}\x00{kind}\x00{reasoning}\x00{content}\x00" + json.dumps(tools, sort_keys = True)
        ).encode("utf-8")
    ).hexdigest()
    return Unit(
        index = index,
        kind = kind,
        reasoning = reasoning,
        content = content,
        chars = chars,
        sha256 = digest,
        tool_calls = tools,
    )


def units_for_chars(total_chars: int, seed: int = CORPUS_SEED) -> list[Unit]:
    """The shortest prefix of the cycle whose characters reach `total_chars`."""
    out: list[Unit] = []
    size = 0
    index = 0
    while size < total_chars:
        u = generate_unit(index, seed)
        out.append(u)
        size += u.chars
        index += 1
        if index > 4096:
            raise RuntimeError("corpus cycle did not reach the requested size")
    return out


def freeze(
    max_chars: int = SHIPPED_CHARS_BUDGET,
    seed: int = CORPUS_SEED,
    out_dir: Path = FROZEN_DIR,
) -> dict:
    """Write `units.jsonl` and `manifest.json`. Run deliberately; never on a benchmark run."""
    out_dir.mkdir(parents = True, exist_ok = True)
    shipped = units_for_chars(max_chars, seed)
    # Sized from the ladder, not a char budget: streamed turns and follow-ups live past the prefix.
    all_units = [generate_unit(i, seed) for i in range(manifest_unit_count(seed))]
    with (out_dir / "units.jsonl").open("w", encoding = "utf-8") as fh:
        for u in shipped:
            fh.write(json.dumps(u.as_row(), ensure_ascii = False) + "\n")
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "seed": seed,
        "prose_chars": PROSE_CHARS,
        "fence_chars": FENCE_CHARS,
        "preamble_fraction": PREAMBLE_FRACTION,
        "math_block_prob": MATH_BLOCK_PROB,
        "inline_math_prob": INLINE_MATH_PROB,
        "max_unit_chars": MAX_UNIT_CHARS,
        "shipped_units": len(shipped),
        "shipped_chars": sum(u.chars for u in shipped),
        "units": [
            {"index": u.index, "kind": u.kind, "chars": u.chars, "sha256": u.sha256}
            for u in all_units
        ],
    }
    manifest["corpus_hash"] = corpus_hash(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent = 2, ensure_ascii = False) + "\n", encoding = "utf-8"
    )
    return manifest


def corpus_hash(manifest: dict) -> str:
    """Includes parameters as well as unit digests, so different declared densities never hash equal."""
    h = hashlib.sha256()
    for key in (
        "corpus_version",
        "seed",
        "prose_chars",
        "fence_chars",
        "preamble_fraction",
        "math_block_prob",
        "inline_math_prob",
        "max_unit_chars",
    ):
        h.update(f"{key}={manifest[key]}\x00".encode("utf-8"))
    for u in manifest["units"]:
        h.update(f"{u['index']}:{u['sha256']}\x00".encode("utf-8"))
    return h.hexdigest()


class Corpus:
    """The frozen corpus, with every unit checked against the manifest before it is used."""

    def __init__(self, manifest: dict, shipped: dict[int, Unit], seed: int) -> None:
        self.manifest = manifest
        self.seed = seed
        self._shipped = shipped
        self._expected = {u["index"]: u["sha256"] for u in manifest["units"]}
        self.corpus_hash = manifest["corpus_hash"]
        self.regenerated: list[int] = []

    @classmethod
    def load(cls, frozen_dir: Optional[Path] = None) -> "Corpus":
        # Use the resource loader: inside studiobench.pyz the corpus is a zip member.
        from ..runtime import resources

        if frozen_dir is not None:
            manifest_path = Path(frozen_dir) / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"no frozen manifest at {manifest_path}")
            raw_manifest = manifest_path.read_text(encoding = "utf-8")
            raw_units = ""
            units_path = Path(frozen_dir) / "units.jsonl"
            if units_path.exists():
                raw_units = units_path.read_text(encoding = "utf-8")
        else:
            try:
                raw_manifest = resources.read_text("fixture/corpus/frozen/manifest.json")
            except (FileNotFoundError, OSError) as exc:
                raise FileNotFoundError(
                    "the frozen corpus is not in this build. Run "
                    "`python -m tests.studio.studiobench.fixture.corpus --freeze` to build it."
                ) from exc
            try:
                raw_units = resources.read_text("fixture/corpus/frozen/units.jsonl")
            except (FileNotFoundError, OSError):
                raw_units = ""
        manifest = json.loads(raw_manifest)
        recomputed = corpus_hash(manifest)
        if recomputed != manifest.get("corpus_hash"):
            raise ValueError(
                f"the frozen manifest's own corpus_hash does not match its contents "
                f"({manifest.get('corpus_hash')} declared, {recomputed} recomputed). The corpus "
                "has been edited by hand; rebuild it with --freeze."
            )
        shipped: dict[int, Unit] = {}
        for line in raw_units.splitlines():
            line = line.strip()
            if line:
                u = Unit.from_row(json.loads(line))
                shipped[u.index] = u
        return cls(manifest, shipped, manifest["seed"])

    def unit(self, index: int) -> Unit:
        """Shipped or regenerated, a unit must match its manifest digest or the generator has drifted."""
        u = self._shipped.get(index)
        if u is None:
            u = generate_unit(index, self.seed)
            self.regenerated.append(index)
        expected = self._expected.get(index)
        if expected is None:
            raise KeyError(f"unit {index} is outside the frozen manifest")
        if u.sha256 != expected:
            raise ValueError(
                f"corpus unit {index} does not match the frozen manifest "
                f"(expected {expected[:16]}, got {u.sha256[:16]}). The generator has drifted; "
                "this tree cannot reproduce the shipped corpus."
            )
        return u

    def units_for_chars(self, total_chars: int) -> list[Unit]:
        out: list[Unit] = []
        size = 0
        index = 0
        while size < total_chars:
            u = self.unit(index)
            out.append(u)
            size += u.chars
            index += 1
        return out

    def iter_units(self) -> Iterator[Unit]:
        for entry in self.manifest["units"]:
            yield self.unit(entry["index"])


RUNGS: dict[str, int] = {
    "1K": 1_000,
    "10K": 10_000,
    "100K": 100_000,
    "500K": 500_000,
    "1M": 1_000_000,
}

# Only sizes the corpus; reported chars_per_token is always the measured one.
PROVISIONAL_CHARS_PER_TOKEN = 4.0

# Stream must finish within ~20 s, before the first after-generation slot at 22.5 s.
STREAM_TAIL_CHARS = 6_000

# Opening turn plus send_turn follow-ups; consecutive units alternate reasoning- and code-heavy.
STREAM_TURNS = 3

FOLLOW_UP_CHARS = 1_500

# Below this, three turns would drain the opening stream too early, so stream once.
MULTI_TURN_MIN_CHARS = 20_000

# Headroom over the provisional 4.0 ratio; above this plan_rung refuses.
MANIFEST_CHARS_PER_TOKEN = 5.0


def manifest_unit_count(
    seed: int = CORPUS_SEED, chars_per_token: float = MANIFEST_CHARS_PER_TOKEN
) -> int:
    """Sized from the ladder: a short manifest once clamped the top rung's streamed turns onto one unit."""
    prefix = len(units_for_chars(int(max(RUNGS.values()) * chars_per_token), seed))
    return prefix + STREAM_TURNS


@dataclass
class RungPlan:
    """What a rung is made of, and how much of it streams rather than being seeded."""

    rung: str
    target_tokens: int
    target_chars: int
    seeded_units: list[Unit] = field(default_factory = list)
    streamed_unit: Optional[Unit] = None
    follow_up_units: list[Unit] = field(default_factory = list)

    @property
    def seeded_chars(self) -> int:
        return sum(u.chars for u in self.seeded_units)

    @property
    def streamed_chars(self) -> int:
        return self.streamed_unit.chars if self.streamed_unit else 0

    @property
    def follow_up_chars(self) -> int:
        return sum(u.chars for u in self.follow_up_units)

    @property
    def total_chars(self) -> int:
        # Follow-ups are on screen, so they count toward the rung's mass.
        return self.seeded_chars + self.streamed_chars + self.follow_up_chars


def dollarise(text: str, salt: str) -> str:
    """Adds non-math dollars to the streamed unit only, not the frozen corpus; exercises the heuristics."""
    lines = text.split("\n")
    out: list[str] = []
    fenced = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            out.append(line)
            continue
        # Shell lines inside fences and prices in prose exercise the code-region and currency paths.
        if fenced and index % 11 == 0:
            out.append(f"{line}  # $HOME/{salt}{index} costs $1{index % 10}.99")
        elif not fenced and line and index % 17 == 0:
            out.append(f"{line} It costs ${index % 9}{index % 10}.99 or ${index % 7},200 a year.")
        else:
            out.append(line)
    return "\n".join(out)


def _too_small(rung: str, chars_per_token: float, last_index: int, what: str) -> ValueError:
    """The one message for every way a rung can outgrow the frozen corpus.

    Deliberately an ERROR and not a clamp. The clamp it replaces is how a 1M rung came to stream
    the same unit three times and report the cheapest per-character cost on the ladder, and a
    benchmark that quietly measures something smaller than it says is worse than one that stops.
    """
    return ValueError(
        f"the frozen corpus is too small for the {rung} rung at {chars_per_token} chars per "
        f"token: {what} runs past the manifest, which ends at index {last_index}. Re-freeze with "
        "`python -m tests.studio.studiobench.fixture.corpus --freeze` after raising "
        "MANIFEST_CHARS_PER_TOKEN, and note that a re-freeze changes corpus_hash and so ends "
        "comparability with everything measured before it."
    )


def _tail_not_deliverable(
    rung: str,
    requested: int,
    source_index: int,
    deliverable: int,
    thread_chars: int,
    target_chars: int,
) -> ValueError:
    """Raises when the corpus cannot deliver the requested tail: a shortfall silently fails to vary it."""
    return ValueError(
        f"the frozen corpus cannot deliver a {requested:,} character streamed tail at the {rung} "
        f"rung: the streamed turn is unit {source_index}, which is {deliverable:,} characters, so "
        f"the reply would be {deliverable / requested:.1%} of the one asked for. Asking for more "
        f"tail makes this worse rather than better, because the seeded prefix shrinks to pay for "
        f"it and the streamed turn moves to a smaller unit -- here the thread would collapse to "
        f"{thread_chars:,} characters against the rung's {target_chars:,}, while still being "
        f"recorded, weighted and reported as {rung}. Ask for less, or move the same request to a "
        f"higher rung. NOT 'at most {deliverable:,}': that is this unit's size at THIS request, "
        f"and the ceiling RISES as the request falls, because a smaller tail leaves a longer "
        f"seeded prefix which draws the streamed turn from a larger unit. Lower the ask until "
        f"this stops firing rather than reading a maximum off this line."
    )


def unit_text(unit: Unit) -> str:
    """Everything a unit puts on screen, in the order it arrives."""
    return unit.reasoning + unit.content


def check_streamed_units_are_new(plan: RungPlan) -> None:
    """Streamed turns must be material the thread has not seen, compared as prefixes of clipped units."""
    streamed: list[tuple[str, str]] = []
    if plan.streamed_unit is not None:
        streamed.append(("streamed", unit_text(plan.streamed_unit)))
    for i, unit in enumerate(plan.follow_up_units):
        streamed.append((f"follow_up[{i}]", unit_text(unit)))

    def fail(what: str, a: str, b: str) -> None:
        raise ValueError(
            f"rung {plan.rung}: {a} and {b} {what}. A streamed turn that repeats material already "
            "on screen hits Shiki's source-keyed cache instead of missing it, so the rung reports "
            "less work per character than it does. The corpus is too small for this ladder; "
            "re-freeze it with `--freeze`."
        )

    seeded = [(f"seeded unit {u.index}", unit_text(u)) for u in plan.seeded_units]
    for i, (label, text) in enumerate(streamed):
        if not text:
            raise ValueError(f"rung {plan.rung}: {label} is empty")
        for other_label, other in streamed[i + 1 :] + seeded:
            if text.startswith(other) or other.startswith(text):
                fail("share a prefix", label, other_label)


def plan_rung(
    corpus: Corpus,
    rung: str,
    chars_per_token: float = PROVISIONAL_CHARS_PER_TOKEN,
    stream_tail_chars: Optional[int] = None,
    dollars: bool = False,
) -> RungPlan:
    """Only the last reply streams, since streaming a million tokens at field cadence would take hours."""
    tokens = RUNGS[rung]
    target_chars = int(tokens * chars_per_token)

    # The streamed tail is constant across rungs so film slots stay correctly labelled; only the
    # seeded prefix grows. So costs scaling with reply length read as a floor on this ladder.
    turns = STREAM_TURNS if target_chars >= MULTI_TURN_MIN_CHARS else 1
    tail_budget = STREAM_TAIL_CHARS if stream_tail_chars is None else max(1, stream_tail_chars)
    tail_target = min(tail_budget, target_chars) if stream_tail_chars is None else tail_budget
    follow_budget = FOLLOW_UP_CHARS * (turns - 1)
    seed_target = max(0, target_chars - tail_target - follow_budget)

    # Trim the prefix's last unit to land on target; whole units would overshoot by a full turn.
    last_index = max(e["index"] for e in corpus.manifest["units"])
    try:
        seeded = corpus.units_for_chars(seed_target) if seed_target > 0 else []
    except KeyError as exc:
        raise _too_small(rung, chars_per_token, last_index, "its seeded prefix alone") from exc
    if seeded:
        overshoot = sum(u.chars for u in seeded) - seed_target
        if overshoot > 0:
            seeded[-1] = seeded[-1].clipped_to(max(1, seeded[-1].chars - overshoot))

    # No clamp: clamping to last_index duplicates the final unit and hits Shiki's cache.
    needed = len(seeded) + turns - 1
    if needed > last_index:
        raise _too_small(
            rung,
            chars_per_token,
            last_index,
            f"its {len(seeded)}-unit seeded prefix plus {turns} streamed turns need units up to "
            f"index {needed}, which",
        )

    # The opening turn keeps the full tail budget; follow-ups are extra.
    source = corpus.unit(len(seeded))
    # An explicit tail must fit in the unit; the default is a ceiling small rungs are allowed under.
    if stream_tail_chars is not None and tail_target >= source.chars:
        raise _tail_not_deliverable(
            rung,
            tail_target,
            len(seeded),
            source.chars,
            sum(u.chars for u in seeded) + source.chars + FOLLOW_UP_CHARS * (turns - 1),
            target_chars,
        )
    streamed = source.clipped_to(tail_target)
    follow_ups = [corpus.unit(len(seeded) + i).clipped_to(FOLLOW_UP_CHARS) for i in range(1, turns)]
    if dollars:
        # Only streamed turns: the seeded prefix is never re-preprocessed.
        streamed = replace(
            streamed,
            reasoning = dollarise(streamed.reasoning, "r"),
            content = dollarise(streamed.content, "c"),
        )
        streamed = replace(streamed, chars = len(streamed.reasoning) + len(streamed.content))
        follow_ups = [
            replace(u, reasoning = dollarise(u.reasoning, "r"), content = dollarise(u.content, "c"))
            for u in follow_ups
        ]
        follow_ups = [replace(u, chars = len(u.reasoning) + len(u.content)) for u in follow_ups]

    plan = RungPlan(
        rung = rung,
        target_tokens = tokens,
        target_chars = target_chars,
        seeded_units = seeded,
        streamed_unit = streamed,
        follow_up_units = follow_ups,
    )
    check_streamed_units_are_new(plan)
    return plan


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description = "Build or verify the frozen studiobench corpus.")
    ap.add_argument("--freeze", action = "store_true", help = "regenerate frozen/*.jsonl")
    ap.add_argument("--verify", action = "store_true", help = "check the tree against the freeze")
    args = ap.parse_args(argv)
    if args.freeze:
        m = freeze()
        print(
            f"froze {m['shipped_units']} units, {m['shipped_chars']:,} chars, "
            f"{len(m['units'])} manifest entries"
        )
        print(f"corpus_hash {m['corpus_hash']}")
        return 0
    c = Corpus.load()
    n = 0
    for _ in c.iter_units():
        n += 1
    print(f"verified {n} units against corpus_hash {c.corpus_hash}")
    print(f"regenerated (not shipped as text): {len(c.regenerated)} units")
    for rung in RUNGS:
        p = plan_rung(c, rung)
        print(
            f"  {rung:>5}: {len(p.seeded_units):>4} seeded units, "
            f"{p.seeded_chars:>10,} seeded chars, {p.streamed_chars:>9,} streamed chars"
        )
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
