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

test("installer progress rotates without making false completion claims", () => {
  assert.equal(installProgressMessage(-1, 0).title, "Preparing your installation...");
  assert.equal(installProgressMessage(2, 0).title, "Setting up your workspace...");
  assert.equal(installProgressMessage(4, 0).title, "Installing required components...");
  assert.equal(installProgressMessage(6, 0).title, "Getting local AI tools ready...");
  assert.equal(installProgressMessage(6, 1).title, "Setup is still working...");
  assert.equal(installProgressMessage(6, 2).title, "Preparing your installation...");

  for (let rotation = 0; rotation < 20; rotation += 1) {
    assert.doesNotMatch(installProgressMessage(999, rotation).title, /nearly done/i);
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
