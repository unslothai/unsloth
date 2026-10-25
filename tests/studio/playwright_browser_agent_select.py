# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

LEAF = (
    Path(__file__).resolve().parents[2] / "studio/frontend/src/features/browser/agent/page-agent.ts"
)
EXECUTOR = LEAF.with_name("executor.ts")
LONG_LABEL = (
    "I agree to receive information about upcoming events and promotional offers from our partners"
)
UNICODE_LABEL = "İletişim bilgilerimin yaklaşan etkinlikler ve iş ortaklarımızın teklifleri hakkında bilgi almak için kullanılmasını kabul ediyorum."


def panel_errors() -> list[re.Pattern[str]]:
    block = re.search(
        r"const AGENT_ERRORS = \[(.*?)\n\];", EXECUTOR.read_text(encoding = "utf-8"), re.S
    ).group(1)
    return [re.compile(body) for body in re.findall(r"^\s*/(.*)/,$", block, re.M)]


def shipped() -> str:
    js = f"import({json.dumps(LEAF.as_uri())}).then((m) => console.log(JSON.stringify(m.pageAgentSource())))"
    cmd = ["node", "--experimental-strip-types", "--no-warnings", "-e", js]
    return json.loads(subprocess.run(cmd, check = True, capture_output = True, text = True).stdout)


@pytest.fixture(scope = "module")
def browser():
    with sync_playwright() as p:
        with getattr(p, os.environ.get("PW_BROWSER", "chromium")).launch(headless = True) as browser:
            yield browser


@pytest.fixture(scope = "module")
def source():
    return shipped()


@pytest.fixture(scope = "module")
def page(browser):
    with browser.new_page(viewport = {"width": 1280, "height": 2400}) as page:
        yield page


@pytest.fixture
def form(page, source):
    def install(html):
        page.set_content(html)
        page.evaluate(source)

        def run(op, **args):
            return page.evaluate(
                "(c) => window[Symbol.for('unsloth.agent')].run(c)", {"op": op, **args}
            )

        text = run("snapshot")["text"]
        refs = {name: ref for ref, name in re.findall(r'\[(r\d+)\] \w+ "([^\"]+)"', text)}
        return page, run, refs

    return install


@pytest.mark.parametrize(
    "html,name,value",
    [
        ('<label><input type="checkbox" value="bacon">Bacon</label>', "Bacon", "  BACON  "),
        (
            '<label for="box">Extra   napkins</label><input id="box" type="checkbox">',
            "Extra napkins",
            " EXTRA  \n NAPKINS ",
        ),
        (
            '<label><input type="checkbox" aria-label="Accessible">Other</label>',
            "Accessible",
            "accessible",
        ),
        (
            '<span id="a">Named</span><span id="b">option</span><input type="checkbox" aria-labelledby="a b">',
            "Named option",
            "named option",
        ),
        ('<label><input type="checkbox" value=" code-7 ">Option</label>', "Option", "CODE-7"),
        (
            '<div role="checkbox" aria-checked="false" onclick="this.setAttribute(\'aria-checked\', \'true\')">Custom</div>',
            "Custom",
            "custom",
        ),
        (f'<label><input type="checkbox">{LONG_LABEL}</label>', LONG_LABEL[:79] + "…", LONG_LABEL),
        (
            f'<label><input type="checkbox">{LONG_LABEL}</label>',
            LONG_LABEL[:79] + "…",
            LONG_LABEL[:79] + "…",
        ),
        (
            f'<label><input type="checkbox">{UNICODE_LABEL}</label>',
            UNICODE_LABEL[:79] + "…",
            UNICODE_LABEL,
        ),
        (
            f'<label><input type="checkbox">{UNICODE_LABEL}</label>',
            UNICODE_LABEL[:79] + "…",
            UNICODE_LABEL[:79] + "…",
        ),
    ],
)
def test_own_label_or_value(form, html, name, value):
    page, run, refs = form(html)
    answer = run("select", ref = refs[name], value = value)
    assert answer["ok"], answer
    if page.locator("input").count():
        assert page.locator("input").is_checked()
    else:
        assert page.locator("[role=checkbox]").get_attribute("aria-checked") == "true"


@pytest.mark.parametrize(
    "word",
    [
        "true",
        "on",
        "yes",
        "checked",
        "1",
        "check",
        "false",
        "off",
        "no",
        "unchecked",
        "0",
        "uncheck",
    ],
)
@pytest.mark.parametrize("role", ["checkbox", "radio", "switch"])
def test_state_words_precede_labels_and_values(form, word, role):
    html = (
        f'<label><input type="{role}" value="{word}">{word}</label>'
        if role != "switch"
        else f"<div role=\"switch\" aria-checked=\"false\" onclick=\"this.setAttribute('aria-checked', String(this.getAttribute('aria-checked') !== 'true'))\">{word}</div>"
    )
    page, run, refs = form(html)
    ref = refs[word]
    expected = word in ("true", "on", "yes", "checked", "1", "check")
    for initial in (False, True):
        page.locator("input, [role=switch]").evaluate(
            "(el, state) => { if (el.tagName === 'INPUT') el.checked = state; else el.setAttribute('aria-checked', String(state)); }",
            initial,
        )
        answer = run("select", ref = ref, value = f" {word.upper()} ")
        state = page.locator("input, [role=switch]").evaluate(
            "el => el.tagName === 'INPUT' ? el.checked : el.getAttribute('aria-checked') === 'true'"
        )
        if role == "radio" and initial and not expected:
            assert not answer["ok"] and state is True, answer
        else:
            assert answer["ok"] and state is expected, answer


@pytest.mark.parametrize("kind", ["native", "aria"])
def test_radio_targets_and_sibling_state_words(form, kind):
    labels = ["Small", "Large", "Yes", "No", "On", "Off", "Checked", "Unchecked", "0", "1"]
    if kind == "native":
        html = '<form id="size"></form>' + "".join(
            f'<label><input type="radio" name="size" form="size" value="{label.lower()}">{label}</label>'
            for label in labels
        )
    else:
        html = (
            '<div role="radiogroup group">'
            + "".join(
                f'<div role="radio option" aria-checked="false" onclick="for (const n of this.parentElement.children) n.setAttribute(\'aria-checked\', String(n === this))">{label}</div>'
                for label in labels
            )
            + "</div>"
        )
    page, run, refs = form(html)

    def selected():
        return page.locator('input:checked, [role="radio option"][aria-checked=true]').evaluate_all(
            "nodes => nodes.map(n => n.tagName === 'INPUT' ? n.value : n.textContent.toLowerCase())"
        )

    for target, other in (("Large", "Small"), ("Small", "Large")):
        assert run("select", ref = refs[target], value = target)["ok"]
        assert selected() == [target.lower()]
        answer = run("select", ref = refs[target], value = other)
        assert not answer["ok"] and selected() == [target.lower()], answer
        assert any(pattern.fullmatch(answer["error"]) for pattern in panel_errors()), answer
        for word in ("yes", "on", "checked", "1", "check", "true"):
            assert run("select", ref = refs[target], value = word)["ok"]
            assert selected() == [target.lower()]
    assert run("select", ref = refs["No"], value = "true")["ok"]
    assert selected() == ["no"]


@pytest.mark.parametrize("value", ["extra", "", "   ", "Other"])
@pytest.mark.parametrize("role", ["checkbox", "radio", "switch"])
def test_unknown_value_fails_without_changing_state(form, value, role):
    html = (
        f'<label><input type="{role}" value="" checked>Option</label>'
        if role != "switch"
        else '<div role="switch" aria-checked="true" onclick="this.setAttribute(\'aria-checked\', \'false\')">Option</div>'
    )
    page, run, refs = form(html)
    answer = run("select", ref = refs["Option"], value = value)
    state = page.locator("input, [role=switch]").evaluate(
        "el => el.tagName === 'INPUT' ? el.checked : el.getAttribute('aria-checked') === 'true'"
    )
    assert answer["ok"] is False and state is True, answer
    assert any(pattern.fullmatch(answer["error"]) for pattern in panel_errors()), answer
    assert '"true", "false"' in answer["error"]


def test_aria_name_takes_precedence_over_html_label(form):
    page, run, refs = form('<label><input type="checkbox" aria-label="Accessible">Other</label>')
    answer = run("select", ref = refs["Accessible"], value = "Other")
    assert answer["ok"] is False and not page.locator("input").is_checked(), answer


def test_dropdown_unchanged(form):
    page, run, refs = form(
        '<select aria-label="Size"><option value="s">Small</option><option value="l">Large</option></select>'
    )
    for value, expected in (("Large", "l"), ("s", "s")):
        assert run("select", ref = refs["Size"], value = value)["ok"]
        assert page.locator("select").input_value() == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--confcutdir", str(Path(__file__).parent)]))
