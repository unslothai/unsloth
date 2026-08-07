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
  assert.equal(models, "Loading application services...");
  assert.equal(startupMessageFromLog(models, "unrelated output"), models);

  const server = startupMessageFromLog(models, "  - Starting server...");
  assert.equal(server, "Starting local server...");
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
    [-1, "Preparing your installation..."],
    [2, "Setting up your workspace..."],
    [4, "Installing required components..."],
    [6, "Getting local AI tools ready..."],
  ]);

  for (const [step, expectedTitle] of expectedTitles) {
    const subtitles = new Set<string>();
    for (let rotation = 0; rotation < 20; rotation += 1) {
      const message = installProgressMessage(step, rotation);
      assert.equal(message.title, expectedTitle);
      assert.doesNotMatch(message.title, /nearly done/i);
      subtitles.add(message.subtitle);
    }
    assert.ok(subtitles.size > 1, `step ${step} should rotate reassurance copy`);
  }
});

test("startup copy rotates while preserving backend phase transitions", () => {
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 0), "Starting Unsloth...");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 1), "Preparing local services...");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 2), "Getting your workspace ready...");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 4), "Preparing local services...");
});

test("status copy rotates every five seconds", () => {
  assert.equal(STATUS_MESSAGE_ROTATION_MS, 5_000);
});
