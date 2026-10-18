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

FORM = """<!doctype html><form>
<fieldset><legend>Pizza Size</legend>
  <p><label><input type="radio" name="size" value="small"> Small </label></p>
  <p><label><input type="radio" name="size" value="large"> Large </label></p>
</fieldset>
<fieldset><legend>Pizza Toppings</legend>
  <p><label><input type="checkbox" name="topping" value="bacon"> Bacon </label></p>
  <p><label><input type="checkbox" name="topping" value="mushroom"> Mushroom </label></p>
  <p><label><input type="checkbox" name="topping" value="onion"> Onion </label></p>
</fieldset>
</form>"""


def shipped() -> str:
    js = f"import({json.dumps(LEAF.as_uri())}).then((m) => console.log(JSON.stringify(m.pageAgentSource())))"
    cmd = ["node", "--experimental-strip-types", "--no-warnings", "-e", js]
    return json.loads(subprocess.run(cmd, check = True, capture_output = True, text = True).stdout)


def main() -> None:
    source = shipped()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = True)
        page = browser.new_page()
        page.set_content(FORM)
        page.evaluate(source)

        def run(command: dict) -> dict:
            return page.evaluate("(c) => window[Symbol.for('unsloth.agent')].run(c)", command)

        def checked(value: str) -> bool:
            return page.is_checked(f"input[value={value}]")

        text = run({"op": "snapshot"})["text"]
        refs = dict((name, ref) for ref, name in re.findall(r"\[(r\d+)\] \w+ \"([^\"]+)\"", text))

        answer = run({"op": "select", "ref": refs["Large"], "value": "Large"})
        assert answer["ok"] and checked("large"), f"a radio named by its own label stayed empty: {answer}"

        answer = run({"op": "select", "ref": refs["Mushroom"], "value": "mushroom"})
        assert answer["ok"] and checked("mushroom"), f"a checkbox named by its label stayed empty: {answer}"

        answer = run({"op": "select", "ref": refs["Bacon"], "value": "bacon"})
        assert answer["ok"] and checked("bacon"), f"a checkbox named by its value stayed empty: {answer}"

        answer = run({"op": "select", "ref": refs["Small"], "value": "Large"})
        assert not answer["ok"] and checked("large") and not checked("small"), (
            f"another option's label was taken as 'uncheck' and reported fine: {answer}"
        )

        answer = run({"op": "select", "ref": refs["Onion"], "value": "extra"})
        assert not answer["ok"] and not checked("onion"), f"an unknown value was reported as done: {answer}"

        assert run({"op": "select", "ref": refs["Onion"], "value": "true"})["ok"] and checked("onion")
        assert run({"op": "select", "ref": refs["Onion"], "value": "false"})["ok"] and not checked("onion")
        browser.close()
    print("[browser-agent-select] radios and checkboxes follow their own labels")


if __name__ == "__main__":
    sys.exit(main())
