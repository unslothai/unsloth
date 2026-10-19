# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

LEAF = (
    Path(__file__).resolve().parents[2] / "studio/frontend/src/features/browser/agent/page-agent.ts"
)
EXECUTOR = LEAF.with_name("executor.ts")

FORM = """<!doctype html><form>
<fieldset><legend>Pizza Size</legend>
  <p><label><input type="radio" name="size" value="small"> Small </label></p>
  <p><label><input type="radio" name="size" value="large"> Large </label></p>
</fieldset>
<fieldset><legend>Extra napkins</legend>
  <p><label><input type="radio" name="napkins" value="yes"> Yes </label></p>
  <p><label><input type="radio" name="napkins" value="no"> No </label></p>
</fieldset>
<fieldset><legend>Pizza Toppings</legend>
  <p><label><input type="checkbox" name="topping" value="bacon"> Bacon </label></p>
  <p><label><input type="checkbox" name="topping" value="mushroom"> Mushroom </label></p>
  <p><label><input type="checkbox" name="topping" value="onion"> Onion </label></p>
</fieldset>
<fieldset><legend>Offers</legend>
  <p><label><input type="checkbox" name="promo" value="false"> Promo emails </label></p>
  <p><label><input type="checkbox" name="caps" value="cb-false"> FALSE </label></p>
</fieldset>
<fieldset><legend>Quiz</legend>
  <p><label><input type="radio" name="tf" value="tf-true"> True </label></p>
  <p><label><input type="radio" name="tf" value="tf-false"> False </label></p>
</fieldset>
<div role="radiogroup" aria-label="Lights">
  <div role="radio" aria-checked="false" tabindex="0" onclick="this.setAttribute('aria-checked', 'true')">On</div>
  <div role="radio" aria-checked="false" tabindex="0" onclick="this.setAttribute('aria-checked', 'true')">Off</div>
</div>
</form>
<div role="radiogroup group" aria-label="Fan">
  <div role="radio option" aria-checked="false" tabindex="0" onclick="this.setAttribute('aria-checked', 'true')">Checked</div>
  <div role="radio option" aria-checked="false" tabindex="0" onclick="this.setAttribute('aria-checked', 'true')">Unchecked</div>
</div>
<form id="quiz"><label><input type="radio" name="quiz" value="zero"> 0 </label></form>
<label><input type="radio" name="quiz" value="one" form="quiz"> 1 </label>"""


def panel_errors() -> list[re.Pattern[str]]:
    block = re.search(
        r"const AGENT_ERRORS = \[(.*?)\n\];", EXECUTOR.read_text(encoding = "utf-8"), re.S
    ).group(1)
    return [re.compile(body) for body in re.findall(r"^\s*/(.*)/,$", block, re.M)]


def shipped() -> str:
    js = f"import({json.dumps(LEAF.as_uri())}).then((m) => console.log(JSON.stringify(m.pageAgentSource())))"
    cmd = ["node", "--experimental-strip-types", "--no-warnings", "-e", js]
    return json.loads(subprocess.run(cmd, check = True, capture_output = True, text = True).stdout)


def main() -> None:
    source = shipped()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = True)
        page = browser.new_page(viewport = {"width": 1280, "height": 2400})
        page.set_content(FORM)
        page.evaluate(source)

        errors: list[str] = []

        def run(command: dict) -> dict:
            answer = page.evaluate("(c) => window[Symbol.for('unsloth.agent')].run(c)", command)
            if answer.get("error"):
                errors.append(answer["error"])
            return answer

        def checked(value: str) -> bool:
            return page.is_checked(f"input[value={value}]")

        text = run({"op": "snapshot"})["text"]
        refs = dict((name, ref) for ref, name in re.findall(r"\[(r\d+)\] \w+ \"([^\"]+)\"", text))

        answer = run({"op": "select", "ref": refs["Large"], "value": "Large"})
        assert answer["ok"] and checked(
            "large"
        ), f"a radio named by its own label stayed empty: {answer}"

        answer = run({"op": "select", "ref": refs["Mushroom"], "value": "mushroom"})
        assert answer["ok"] and checked(
            "mushroom"
        ), f"a checkbox named by its label stayed empty: {answer}"

        answer = run({"op": "select", "ref": refs["Bacon"], "value": "bacon"})
        assert answer["ok"] and checked(
            "bacon"
        ), f"a checkbox named by its value stayed empty: {answer}"

        answer = run({"op": "select", "ref": refs["Small"], "value": "Large"})
        assert (
            not answer["ok"] and checked("large") and not checked("small")
        ), f"another option's label was taken as 'uncheck' and reported fine: {answer}"

        answer = run({"op": "select", "ref": refs["Onion"], "value": "extra"})
        assert not answer["ok"] and not checked(
            "onion"
        ), f"an unknown value was reported as done: {answer}"

        answer = run({"op": "select", "ref": refs["Yes"], "value": "No"})
        assert (
            not answer["ok"] and not checked("yes") and not checked("no")
        ), f"a sibling's label that is an off word was taken as 'uncheck': {answer}"

        assert run({"op": "select", "ref": refs["No"], "value": "No"})["ok"] and checked("no")

        answer = run({"op": "select", "ref": refs["No"], "value": "Yes"})
        assert (
            not answer["ok"] and checked("no") and not checked("yes")
        ), f"a sibling's label that is an on word was taken as 'check': {answer}"

        assert run({"op": "select", "ref": refs["Onion"], "value": "true"})["ok"] and checked(
            "onion"
        )
        assert run({"op": "select", "ref": refs["Onion"], "value": "false"})["ok"] and not checked(
            "onion"
        )
        assert run({"op": "select", "ref": refs["Promo emails"], "value": "true"})[
            "ok"
        ] and checked("false")
        answer = run({"op": "select", "ref": refs["Promo emails"], "value": "false"})
        assert answer["ok"] and not checked(
            "false"
        ), f"'false' on a box whose HTML value is 'false' did not uncheck it: {answer}"

        answer = run({"op": "select", "ref": refs["Off"], "value": "On"})
        lights = [
            page.locator("[role=radio]", has_text = name).get_attribute("aria-checked")
            for name in ("On", "Off")
        ]
        assert not answer["ok"] and lights == [
            "false",
            "false",
        ], f"an ARIA sibling's label that is an on word was taken as 'check': {answer}"

        answer = run({"op": "select", "ref": refs["Unchecked"], "value": "Checked"})
        fan = [
            page.get_by_text(name, exact = True).get_attribute("aria-checked")
            for name in ("Checked", "Unchecked")
        ]
        assert not answer["ok"] and fan == [
            "false",
            "false",
        ], f"a fallback-role sibling's label that is an on word was taken as 'check': {answer}"

        answer = run({"op": "select", "ref": refs["0"], "value": "1"})
        assert (
            not answer["ok"] and not checked("zero") and not checked("one")
        ), f"a sibling outside the form was missed and its label taken as 'check': {answer}"

        answer = run({"op": "select", "ref": refs["False"], "value": "false"})
        assert answer["ok"] and not checked(
            "tf-false"
        ), f"'false' on a radio labelled False checked it: {answer}"
        answer = run({"op": "select", "ref": refs["False"], "value": "true"})
        assert (
            answer["ok"] and checked("tf-false") and not checked("tf-true")
        ), f"'true' on the False radio was not obeyed: {answer}"

        for value, state in (("false", False), ("true", True), ("false", False)):
            answer = run({"op": "select", "ref": refs["FALSE"], "value": value})
            assert (
                answer["ok"] and checked("cb-false") is state
            ), f"{value!r} on a checkbox labelled FALSE was not obeyed: {answer}"

        unknown = [error for error in errors if not any(p.fullmatch(error) for p in panel_errors())]
        assert not unknown, f"errors the panel would hide from the model: {unknown}"
        browser.close()
    print("[browser-agent-select] radios and checkboxes follow their own labels")


if __name__ == "__main__":
    sys.exit(main())
