# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path
import json
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
    config = tmp_path / "subagent.json"
    config.write_text(
        json.dumps(
            {
                "baseUrl": "http://127.0.0.1:8000/v1",
                "apiKey": "private-token",
                "model": "local-model",
                "contextWindow": 32768,
                "maxTokens": 8192,
            }
        ),
        encoding = "utf-8",
    )
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
    process.env.UNSLOTH_PI_SUBAGENT_CONFIG = {str(config)!r};
    process.env.PI_CHILD_READY = {str(ready)!r};
    process.env.PI_CANCEL_MARKER = {str(marker)!r};
    process.argv[1] = {str(driver)!r};

    const loaded = await import(pathToFileURL({str(extension)!r}).href);
    let tool;
    let provider;
    loaded.default({{
        registerProvider(_name, value) {{ provider = value; }},
        registerTool(value) {{ tool = value; }},
    }});
    expect(process.env.UNSLOTH_PI_SUBAGENT_CONFIG).toBeUndefined();
    expect(process.env.UNSLOTH_PI_SUBAGENT_API_KEY).toBeUndefined();
    expect(provider.apiKey).toBe("private-token");

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


@pytest.mark.skipif(os.name == "nt", reason = "POSIX driver script")
def test_pi_child_error_events_fail_the_tool_call(tmp_path):
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("Bun is required to execute the bundled Pi extension")

    config = tmp_path / "subagent.json"
    config.write_text(
        json.dumps(
            {
                "baseUrl": "http://127.0.0.1:8000/v1",
                "apiKey": "private-token",
                "model": "local-model",
                "contextWindow": 32768,
                "maxTokens": 8192,
            }
        ),
        encoding = "utf-8",
    )
    # Pi reports model/API failures as message_end events while exiting 0.
    driver = tmp_path / "pi-driver.js"
    driver.write_text(
        """
const event = {
    type: "message_end",
    message: { role: "assistant", stopReason: "error", errorMessage: "backend unreachable", content: [] },
};
console.log(JSON.stringify(event));
""",
        encoding = "utf-8",
    )
    extension = Path(__file__).parents[1] / "pi_subagent.ts"
    test_file = tmp_path / "pi-error.test.ts"
    test_file.write_text(
        f"""
import {{ expect, mock, test }} from "bun:test";
import {{ pathToFileURL }} from "node:url";

mock.module("typebox", () => ({{
    Type: {{ Object: (value) => value, String: (value) => value }},
}}));

test("child error events fail the tool call", async () => {{
    process.env.UNSLOTH_PI_SUBAGENT_CONFIG = {str(config)!r};
    process.argv[1] = {str(driver)!r};

    const loaded = await import(pathToFileURL({str(extension)!r}).href);
    let tool;
    loaded.default({{
        registerProvider() {{}},
        registerTool(value) {{ tool = value; }},
    }});

    const execution = tool.execute(
        "call",
        {{ task: "fail" }},
        undefined,
        undefined,
        {{ cwd: {str(tmp_path)!r} }},
    );
    await expect(execution).rejects.toThrow("backend unreachable");
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
