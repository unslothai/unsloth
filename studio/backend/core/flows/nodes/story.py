# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Story Game nodes that run in the engine. Scene/Choice/... are authored for the player and compile in core/flows/story.py."""

from __future__ import annotations

import copy
from typing import Any

from core.flows.expr import seed_from
from core.flows.registry import FlowNodeError, NodeContext, handler
from core.flows.schema import StoryManifest
from core.flows.story import (
    StoryCompileError,
    finalize,
    repair_generated,
    repair_live_scene,
    scene_schema,
    story_schema,
)


def _size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    try:
        w, h = str(value).lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return default


def _text(ctx: NodeContext, port: str, param: str) -> str:
    v = ctx.inputs.get(port)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return str(ctx.params.get(param) or "").strip()


def _story_input(ctx: NodeContext) -> dict:
    story = ctx.inputs.get("story")
    if not isinstance(story, dict):
        raise FlowNodeError("Connect a story (from a Story Generator or another story node).")
    try:
        return StoryManifest.model_validate(story).model_dump(by_alias = True, exclude_none = True)
    except Exception as exc:  # noqa: BLE001 - pydantic detail is long; the first line is enough
        raise FlowNodeError(
            f"The connected story is not valid: {str(exc).splitlines()[0]}"
        ) from exc


GENERATOR_SYSTEM = """You are a narrative designer writing a short branching visual novel.
Write vivid but brief dialogue (1-2 sentences per line). Every scene id is short snake_case.
`speaker` is a character id, or "narrator". `goto` must be the id of another scene you wrote.
Non-ending scenes have 2-3 choices; ending scenes have `is_ending` true, no choices, an `ending_title` and an `ending_kind`.
Choices can change numeric variables with `effects` ({"var": "courage", "add": 1}).
`background_prompt` describes the place only (no people, no text), for an illustrator.
`look` describes a character's appearance for a portrait artist."""


@handler("story.generator")
async def story_generator(ctx: NodeContext) -> dict:
    p = ctx.params
    premise = _text(ctx, "premise", "premise")
    if not premise:
        raise FlowNodeError("Give the Story Generator a premise.")
    scenes, chars, endings = (
        int(p.get("scenes") or 8),
        int(p.get("characters") or 3),
        int(p.get("endings") or 3),
    )
    prompt = (
        f"Premise: {premise}\nTone: {p.get('tone') or 'whimsical'}\n"
        f"Write {scenes} scenes in total, including exactly {endings} different endings, with {chars} characters.\n"
        f"Start at the scene named in `start`. Make at least one ending good and one surprising.\n"
        f"Variation seed: {ctx.seed}"
    )
    ctx.progress(0.05, phase = "writing")
    raw = await ctx.llm_json(
        model = p.get("model"),
        system = GENERATOR_SYSTEM,
        prompt = prompt,
        schema = story_schema(scenes, chars),
        temperature = 0.9,
        max_tokens = min(1200 + 700 * scenes, 16000),
        seed = ctx.seed,
    )
    try:
        story = repair_generated(
            raw,
            endings = endings,
            seed = ctx.seed,
            style = p.get("style") or None,
            mode = p.get("mode") or "live",
        )
    except StoryCompileError as exc:
        raise FlowNodeError(f"The model's story could not be used: {exc}", retryable = True) from exc
    ctx.log(
        f"{story['title']}: {len(story['scenes'])} scenes, {len(story['characters'])} characters"
    )
    return {"story": story}


def _with_style(
    prompt: str,
    style: Any,
    extra: str = "",
) -> str:
    parts = [prompt.strip().rstrip(".")]
    if style:
        parts.append(str(style))
    if extra:
        parts.append(extra)
    return ", ".join(x for x in parts if x)


@handler("story.illustrate")
async def story_illustrate(ctx: NodeContext) -> dict:
    p = ctx.params
    story = copy.deepcopy(_story_input(ctx))
    style = story.get("style") or ""
    bw, bh = _size(p.get("background_size"), (1344, 768))
    pw, ph = _size(p.get("portrait_size"), (768, 1024))
    todo_bg = [
        s
        for s in story["scenes"]
        if not s.get("background") and (s.get("background_prompt") or s.get("title"))
    ]
    todo_pt = (
        [c for c in story["characters"] if not c.get("portrait")]
        if p.get("portraits", True)
        else []
    )
    total = max(1, len(todo_bg) + len(todo_pt))
    done = 0
    # Scenes that share a prompt share one painting.
    painted: dict[str, dict] = {}
    for s in todo_bg:
        ctx.check_cancelled()
        prompt = _with_style(
            s.get("background_prompt") or s.get("title") or "",
            style,
            "wide establishing shot, no people, no text, no watermark",
        )
        if prompt not in painted:
            ctx.progress(
                done / total,
                phase = f"background: {s.get('title') or s['id']}",
                step = done,
                total = total,
            )
            imgs = await ctx.generate_image(
                model = p.get("model"),
                prompt = prompt,
                width = bw,
                height = bh,
                seed = seed_from(ctx.seed, "bg", s["id"]),
            )
            if not imgs:
                raise FlowNodeError(f"No image came back for scene {s['id']}")
            painted[prompt] = imgs[0]
        s["background"] = painted[prompt]
        done += 1
    for c in todo_pt:
        ctx.check_cancelled()
        ctx.progress(done / total, phase = f"portrait: {c['name']}", step = done, total = total)
        prompt = _with_style(
            f"character portrait of {c['name']}, {c.get('look') or ''}",
            style,
            "upper body, facing the viewer, centered, plain light grey studio background",
        )
        imgs = await ctx.generate_image(
            model = p.get("model"),
            prompt = prompt,
            width = pw,
            height = ph,
            seed = seed_from(ctx.seed, "char", c["id"]),
        )
        if not imgs:
            raise FlowNodeError(f"No portrait came back for {c['name']}")
        portrait = imgs[0]
        if p.get("remove_bg", True):
            try:
                portrait, _mask = await ctx.remove_background(
                    portrait, f"the character {c['name']}", trim = True
                )
            except FlowNodeError as exc:
                ctx.log(f"Kept {c['name']}'s background: {exc}", "warning")
        c["portrait"] = portrait
        done += 1
    ctx.progress(1.0, phase = "done", step = total, total = total)
    return {"story": finalize(story, [])}


@handler("story.narrate")
async def story_narrate(ctx: NodeContext) -> dict:
    p = ctx.params
    story = copy.deepcopy(_story_input(ctx))
    voices = {c["id"]: c.get("voice") or "" for c in story["characters"]}
    lines = [
        ln
        for s in story["scenes"]
        for ln in s.get("lines") or []
        if not ln.get("voice") and ln.get("text")
    ]
    total = max(1, len(lines))
    cache: dict[tuple[str, str], dict] = {}
    for i, ln in enumerate(lines):
        ctx.check_cancelled()
        ctx.progress(i / total, phase = "narrating", step = i, total = total)
        voice = voices.get(ln.get("speaker") or "", "") or str(p.get("narrator_voice") or "")
        key = (voice, ln["text"])
        if key not in cache:
            cache[key] = await ctx.speak(
                model = p.get("model"), text = ln["text"], voice = voice, seed = seed_from(ctx.seed, voice)
            )
        ln["voice"] = cache[key]
    ctx.progress(1.0, phase = "done", step = total, total = total)
    return {"story": finalize(story, [])}


@handler("story.compile")
async def story_compile(ctx: NodeContext) -> dict:
    if isinstance(ctx.inputs.get("story"), dict):
        return {"story": finalize(copy.deepcopy(_story_input(ctx)), [])}
    # Unconnected: the authored Story/Scene graph compiles when the player opens (GET /story).
    ctx.log("The authored Story and Scene nodes compile when you press Play.")
    return {}


NEXT_SCENE_SYSTEM = """You continue an interactive story one scene at a time.
Write the next scene as JSON: a short title, a `background_prompt` (the place only, no people, no text),
2-6 lines of dialogue (`speaker` is a character id or "narrator"), and {choices} choices for the player.
Choices may change numeric variables with `effects`. Set `is_ending` only when the story has truly concluded
(after at least 8 scenes, or when a choice makes an ending inevitable); then give no choices, an `ending_title` and an `ending_kind`."""


@handler("story.next_scene")
async def story_next_scene(ctx: NodeContext) -> dict:
    state = ctx.inputs.get("state")
    if not isinstance(state, dict):
        state = {}
    story = state.get("story") if isinstance(state.get("story"), dict) else {}
    choices = int(ctx.params.get("choices") or 3)
    chars = ", ".join(
        f"{c.get('name')} (id {c.get('id')}): {c.get('look') or ''}"
        for c in story.get("characters") or []
    )
    history = [str(h)[:300] for h in (state.get("history") or [])][-12:]
    prompt = (
        f"Story: {story.get('title') or 'Untitled'}\nCharacters: {chars or 'none yet'}\n"
        f"Variables: {state.get('vars') or {}}\n"
        f"What happened so far (latest last):\n" + ("\n".join(history) or "(the beginning)") + "\n"
        f"The player chose: {state.get('choice') or '(start)'}\n"
        f"{ctx.params.get('instructions') or ''}"
    )
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = NEXT_SCENE_SYSTEM.replace("{choices}", str(choices)),
        prompt = prompt,
        schema = scene_schema(choices),
        temperature = 0.9,
        max_tokens = 1500,
        seed = seed_from(ctx.seed, len(history), state.get("choice")),
    )
    try:
        return {"scene": repair_live_scene(raw, state, choices = choices)}
    except StoryCompileError as exc:
        raise FlowNodeError(str(exc), retryable = True) from exc


@handler("story.scene_art")
async def story_scene_art(ctx: NodeContext) -> dict:
    prompt = _text(ctx, "prompt", "prompt")
    if not prompt:
        raise FlowNodeError("Scene Art needs a prompt.")
    w, h = _size(ctx.params.get("size"), (1344, 768))
    imgs = await ctx.generate_image(
        model = ctx.params.get("model"),
        prompt = _with_style(prompt, ctx.params.get("style"), "no text, no watermark"),
        width = w,
        height = h,
        seed = seed_from(ctx.seed, prompt),
    )
    if not imgs:
        raise FlowNodeError("No image came back")
    return {"image": imgs[0]}
