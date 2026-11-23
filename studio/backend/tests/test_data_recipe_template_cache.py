# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest

pytest.importorskip("data_designer")

import jinja2
from data_designer.engine.processing.ginja.environment import UserTemplateSandboxEnvironment
from data_designer.engine.processing.ginja.exceptions import UserTemplateError

from core.data_recipe import service

TEMPLATE = (
    "Review {{ name }} ({{ category | lower }}) as a {{ job | title }}. "
    "{% if rating > 3 %}Liked it: {{ rating }}/5.{% else %}Disliked it: {{ rating }}/5.{% endif %} "
    "Features: {% for f in features %}{{ f }}{% if not loop.last %}, {% endif %}{% endfor %}. "
    "Note: {{ note | trim }}"
)
RECORD = {
    "name": "Kettle",
    "category": "KITCHEN",
    "job": "nurse",
    "rating": 4,
    "features": ["fast", "quiet"],
    "note": "  ok  ",
}


@pytest.fixture(name = "cache_patch")
def fixture_cache_patch(monkeypatch):
    monkeypatch.setattr(
        UserTemplateSandboxEnvironment, "from_string", jinja2.Environment.from_string
    )
    monkeypatch.setattr(
        UserTemplateSandboxEnvironment,
        "_unsloth_template_cache_patch_applied",
        False,
        raising = False,
    )
    return monkeypatch


def _count_compiles(env, monkeypatch) -> list:
    calls = []
    original = env.compile

    def counting(*args, **kwargs):
        calls.append(args[0] if args else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "compile", counting, raising = False)
    return calls


def _reference_render(env, template: str, record: dict) -> str:
    return jinja2.Environment.from_string(env, template).render(record)


def test_second_render_reuses_the_compiled_template(cache_patch) -> None:
    service._apply_data_designer_template_cache_patch()
    env = UserTemplateSandboxEnvironment(allowed_references = list(RECORD))
    compiles = _count_compiles(env, cache_patch)

    first = env.safe_render(TEMPLATE, RECORD, skip_template_validation = True)
    for _ in range(20):
        assert env.safe_render(TEMPLATE, RECORD, skip_template_validation = True) == first

    assert len(compiles) == 1
    assert first == _reference_render(env, TEMPLATE, RECORD)
    assert "Liked it: 4/5" in first and "Nurse" in first and "kitchen" in first


def test_cache_is_per_environment(cache_patch) -> None:
    service._apply_data_designer_template_cache_patch()
    a = UserTemplateSandboxEnvironment(allowed_references = list(RECORD))
    b = UserTemplateSandboxEnvironment(allowed_references = list(RECORD))
    a.safe_render(TEMPLATE, RECORD, skip_template_validation = True)
    compiles = _count_compiles(b, cache_patch)
    b.safe_render(TEMPLATE, RECORD, skip_template_validation = True)
    assert len(compiles) == 1


def test_rendered_text_matches_an_uncached_compile_across_records(cache_patch) -> None:
    service._apply_data_designer_template_cache_patch()
    env = UserTemplateSandboxEnvironment(allowed_references = list(RECORD))
    for rating in range(1, 6):
        record = {**RECORD, "rating": rating, "name": f"Item {rating}"}
        assert env.safe_render(
            TEMPLATE, record, skip_template_validation = True
        ) == _reference_render(env, TEMPLATE, record)


def test_studio_finalize_and_wrapped_filters_still_apply_on_cache_hits(cache_patch) -> None:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
    )
    from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe

    service._apply_data_designer_prompt_blank_patch()
    service._apply_data_designer_template_cache_patch()
    renderer = RecordBasedPromptRenderer(TextResponseRecipe())

    def render(value):
        return renderer.render(
            prompt_template = "<{{ a }}|{{ a | lower }}|{{ a | trim }}>",
            record = {"a": value},
            prompt_type = PromptType.USER_PROMPT,
        )

    for _ in range(3):
        assert render(None) == "<||>"
        assert render(float("nan")) == "<||>"
        assert render(" Ab ") == "< Ab | ab |Ab>"
    env = renderer._render_func_registry[PromptType.USER_PROMPT.value].func.__self__
    assert len(env.__dict__[service._TEMPLATE_CACHE_ATTR]) == 1


def test_finalize_change_recompiles(cache_patch) -> None:
    service._apply_data_designer_template_cache_patch()
    env = UserTemplateSandboxEnvironment(allowed_references = ["a"])
    assert env.safe_render("[{{ a }}]", {"a": None}, skip_template_validation = True) == "[None]"
    env.finalize = service._blank_missing_prompt_value
    assert env.safe_render("[{{ a }}]", {"a": None}, skip_template_validation = True) == "[]"


def test_filter_change_recompiles(cache_patch) -> None:
    # Compiled code bakes in filters, so a swapped filter must not reuse the old template.
    service._apply_data_designer_template_cache_patch()
    env = UserTemplateSandboxEnvironment(allowed_references = ["a"])
    folded = "{{ 'ABC' | lower }}"
    by_value = "{{ a | lower }}"
    assert env.safe_render(folded, {"a": "X"}, skip_template_validation = True) == "abc"
    assert env.safe_render(by_value, {"a": "X"}, skip_template_validation = True) == "x"

    env.filters["lower"] = jinja2.pass_environment(lambda _env, value: f"swapped {value}")

    assert env.safe_render(folded, {"a": "X"}, skip_template_validation = True) == "swapped ABC"
    assert env.safe_render(by_value, {"a": "X"}, skip_template_validation = True) == "swapped X"
    assert env.safe_render(
        by_value, {"a": "X"}, skip_template_validation = True
    ) == _reference_render(env, by_value, {"a": "X"})


def test_failed_compile_is_not_cached_and_errors_match(cache_patch) -> None:
    template = "{{ a | nosuchfilter }}"
    plain = UserTemplateSandboxEnvironment(allowed_references = ["a"])
    with pytest.raises(Exception) as before:
        plain.safe_render(template, {"a": 1}, skip_template_validation = True)

    service._apply_data_designer_template_cache_patch()
    env = UserTemplateSandboxEnvironment(allowed_references = ["a"])
    compiles = _count_compiles(env, cache_patch)
    errors = []
    for _ in range(2):
        with pytest.raises(Exception) as caught:
            env.safe_render(template, {"a": 1}, skip_template_validation = True)
        errors.append(caught.value)

    assert len(compiles) == 2
    assert not env.__dict__.get(service._TEMPLATE_CACHE_ATTR)
    for error in errors:
        assert type(error) is type(before.value)
        assert str(error) == str(before.value)


def test_empty_render_still_raises_on_every_row(cache_patch) -> None:
    service._apply_data_designer_template_cache_patch()
    env = UserTemplateSandboxEnvironment(allowed_references = ["a"])
    for _ in range(2):
        with pytest.raises(UserTemplateError, match = "renders to empty text"):
            env.safe_render("{{ a }}", {"a": ""}, skip_template_validation = True)


def test_guard_leaves_an_overridden_from_string_untouched(cache_patch) -> None:
    def custom(
        self,
        source,
        globals = None,
        template_class = None,
    ):
        return jinja2.Environment.from_string(self, source, globals, template_class)

    cache_patch.setattr(UserTemplateSandboxEnvironment, "from_string", custom)
    service._apply_data_designer_template_cache_patch()
    assert UserTemplateSandboxEnvironment.from_string is custom
    assert not UserTemplateSandboxEnvironment._unsloth_template_cache_patch_applied


def test_guard_leaves_an_unexpected_signature_untouched(cache_patch) -> None:
    original = jinja2.Environment.from_string

    def other_shape(
        self,
        source,
        *,
        globals = None,
    ):
        return original(self, source, globals)

    cache_patch.setattr(jinja2.Environment, "from_string", other_shape)
    cache_patch.setattr(UserTemplateSandboxEnvironment, "from_string", other_shape)
    service._apply_data_designer_template_cache_patch()
    assert UserTemplateSandboxEnvironment.from_string is other_shape
    assert not UserTemplateSandboxEnvironment._unsloth_template_cache_patch_applied


def test_guard_leaves_a_class_without_safe_render_untouched(cache_patch) -> None:
    cache_patch.setattr(UserTemplateSandboxEnvironment, "safe_render", None)
    service._apply_data_designer_template_cache_patch()
    assert UserTemplateSandboxEnvironment.from_string is jinja2.Environment.from_string
    assert not UserTemplateSandboxEnvironment._unsloth_template_cache_patch_applied


def test_patch_is_idempotent(cache_patch) -> None:
    service._apply_data_designer_template_cache_patch()
    patched = UserTemplateSandboxEnvironment.from_string
    service._apply_data_designer_template_cache_patch()
    assert UserTemplateSandboxEnvironment.from_string is patched
    assert patched is not jinja2.Environment.from_string


def test_create_data_designer_applies_the_cache(cache_patch, tmp_path: Path) -> None:
    service.create_data_designer({"columns": []}, artifact_path = str(tmp_path / "artifacts"))
    assert UserTemplateSandboxEnvironment._unsloth_template_cache_patch_applied
    assert UserTemplateSandboxEnvironment.from_string is not jinja2.Environment.from_string
