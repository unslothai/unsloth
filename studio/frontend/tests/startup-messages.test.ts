import assert from "node:assert/strict";
import test from "node:test";

import {
  INITIAL_STARTUP_MESSAGE,
  installProgressMessage,
  STATUS_MESSAGE_ROTATION_MS,
  UPDATE_STARTUP_MESSAGE,
  startupMessageFromLog,
  startupWaitingMessage,
} from "../src/components/tauri/startup-messages.ts";

test("startup messages follow backend phases without regressing", () => {
  const models = startupMessageFromLog(
    INITIAL_STARTUP_MESSAGE,
    "  - loading PyTorch, Unsloth and Transformers...",
  );
  assert.equal(models, "desktopStartup.loadingModels");
  assert.equal(startupMessageFromLog(models, "unrelated output"), models);

  const server = startupMessageFromLog(models, "  - Starting server...");
  assert.equal(server, "desktopStartup.nearlyDone");
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
    [-1, "desktopStartup.install.preparing"],
    [2, "desktopStartup.install.downloading"],
    [4, "desktopStartup.install.installing"],
    [6, "desktopStartup.install.finishing"],
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
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 0), "desktopStartup.starting");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 1), "desktopStartup.loadingProjects");
  assert.equal(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 2), "desktopStartup.starting");
});

test("nearly done only appears after the backend starts its server", () => {
  const models = startupMessageFromLog(
    INITIAL_STARTUP_MESSAGE,
    "  - loading PyTorch, Unsloth and Transformers...",
  );
  const server = startupMessageFromLog(models, "  - Starting server...");
  assert.notEqual(startupWaitingMessage(INITIAL_STARTUP_MESSAGE, 20), "desktopStartup.nearlyDone");
  assert.notEqual(startupWaitingMessage(models, 20), "desktopStartup.nearlyDone");
  assert.equal(startupWaitingMessage(server, 20), "desktopStartup.nearlyDone");
});

test("status copy rotates every five seconds", () => {
  assert.equal(STATUS_MESSAGE_ROTATION_MS, 5_000);
});

test("the update wait keeps its copy until the backend starts", () => {
  assert.equal(startupWaitingMessage(UPDATE_STARTUP_MESSAGE, 3), "desktopStartup.finishingUpdate");
  assert.equal(
    startupMessageFromLog(UPDATE_STARTUP_MESSAGE, "unrelated output"),
    UPDATE_STARTUP_MESSAGE,
  );
  assert.equal(
    startupMessageFromLog(UPDATE_STARTUP_MESSAGE, "  - Starting server..."),
    "desktopStartup.nearlyDone",
  );
});
