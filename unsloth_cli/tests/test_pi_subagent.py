# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.name == "nt", reason = "POSIX process group regression test")
def test_pi_cancel_kills_child_process_group(tmp_path):
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("Bun is required to execute the bundled Pi extension")

    ready = tmp_path / "grandchild-ready"
    marker = tmp_path / "grandchild-survived"
    driver = tmp_path / "pi-driver.js"
    driver.write_text(
        """
import { spawn } from "node:child_process";

spawn(
    process.execPath,
    [
        "-e",
        `
            const fs = require("node:fs");
            process.on("SIGTERM", () => {});
            fs.writeFileSync(process.env.PI_CHILD_READY, "ready");
            setTimeout(() => fs.writeFileSync(process.env.PI_CANCEL_MARKER, "alive"), 3000);
            setInterval(() => {}, 1000);
        `,
    ],
    { stdio: "inherit" },
);
process.on("SIGTERM", () => {});
setInterval(() => {}, 1000);
""",
        encoding = "utf-8",
    )
    extension = Path(__file__).parents[1] / "pi_subagent.ts"
    test_file = tmp_path / "pi-cancel.test.ts"
    test_file.write_text(
        f"""
import {{ expect, mock, test }} from "bun:test";
import {{ existsSync }} from "node:fs";
import {{ pathToFileURL }} from "node:url";

mock.module("typebox", () => ({{
    Type: {{ Object: (value) => value, String: (value) => value }},
}}));

test("cancellation stops the Pi child process group", async () => {{
    process.env.UNSLOTH_PI_SUBAGENT_MODEL = "local-model";
    process.env.UNSLOTH_PI_SUBAGENT_BASE_URL = "http://127.0.0.1:8000/v1";
    process.env.PI_CHILD_READY = {str(ready)!r};
    process.env.PI_CANCEL_MARKER = {str(marker)!r};
    process.argv[1] = {str(driver)!r};

    const loaded = await import(pathToFileURL({str(extension)!r}).href);
    let tool;
    loaded.default({{
        registerProvider() {{}},
        registerTool(value) {{ tool = value; }},
    }});

    const controller = new AbortController();
    const execution = tool.execute(
        "call",
        {{ task: "wait" }},
        controller.signal,
        undefined,
        {{ cwd: {str(tmp_path)!r} }},
    );
    for (let attempt = 0; attempt < 100 && !existsSync({str(ready)!r}); attempt++) {{
        await Bun.sleep(20);
    }}
    expect(existsSync({str(ready)!r})).toBe(true);
    controller.abort();
    await expect(execution).rejects.toThrow("cancelled");
    await Bun.sleep(3200);
    expect(existsSync({str(marker)!r})).toBe(false);
}}, 10_000);
""",
        encoding = "utf-8",
    )

    completed = subprocess.run(
        [bun, "test", str(test_file)],
        capture_output = True,
        text = True,
        timeout = 15,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
