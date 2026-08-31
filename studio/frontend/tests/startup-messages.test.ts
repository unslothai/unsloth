import assert from "node:assert/strict";
import test from "node:test";

import {
  INITIAL_STARTUP_MESSAGE,
  installProgressMessage,
  STATUS_MESSAGE_ROTATION_MS,
  startupMessageFromLog,
  startupWaitingMessage,
} from "../src/components/tauri/startup-messages.ts";

test("startup messages follow backend phases without regressing", () => {
  const models = startupMessageFromLog(
    INITIAL_STARTUP_MESSAGE,
    "  - loading PyTorch, Unsloth and Transformers...",
  );
  assert.equal(models, "Loading models...");
  assert.equal(startupMessageFromLog(models, "unrelated output"), models);

  const server = startupMessageFromLog(models, "  - Starting server...");
  assert.equal(server, "Nearly done...");
  assert.equal(
    startupMessageFromLog(
      server,
      "  - loading PyTorch, Unsloth and Transformers...",
    ),
    server,
  );
});

test("installer progress rotates reassurance without changing actual phases", () => {
  const expectedTitles = new Map([
    [-1, "Preparing your workspace..."],
    [2, "Downloading required components..."],
    [4, "Installing Unsloth..."],
    [6, "Finishing setup..."],
  ]);

  for (const [step, expectedTitle] of expectedTitles) {
    const subtitles = new Set<string>();
    for (let rotation = 0; rotation < 20; rotation += 1) {
      const message = installProgressMessage(step, rotation);
      assert.equal(message.title, expectedTitle);
      subtitles.add(message.subtitle);
    }
    assert.ok(subtitles.size > 1, `step ${step} should rotate reassurance copy`);
  }
});

test("startup copy rotates while preserving backend phase transitions", () => {
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 0), "Starting Unsloth...");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 1), "Loading projects...");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 2), "Starting Unsloth...");
});

test("nearly done only appears after the backend starts its server", () => {
  const models = startupMessageFromLog(
    INITIAL_STARTUP_MESSAGE,
    "  - loading PyTorch, Unsloth and Transformers...",
  );
  const server = startupMessageFromLog(models, "  - Starting server...");
  assert.notEqual(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 20), "Nearly done...");
  assert.notEqual(startupWaitingMessage(models, 20), "Nearly done...");
  assert.equal(startupWaitingMessage(server, 20), "Nearly done...");
});

test("status copy rotates every five seconds", () => {
  assert.equal(STATUS_MESSAGE_ROTATION_MS, 5_000);
});
