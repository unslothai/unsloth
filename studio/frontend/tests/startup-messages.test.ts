import assert from "node:assert/strict";
import test from "node:test";

import {
  INITIAL_STARTUP_MESSAGE,
  SERVER_START_FALLBACK_MS,
  startupMessageFromLog,
} from "../src/components/tauri/startup-messages.ts";

test("startup messages follow backend phases without regressing", () => {
  const models = startupMessageFromLog(
    INITIAL_STARTUP_MESSAGE,
    "  - loading PyTorch, Unsloth and Transformers...",
  );
  assert.equal(models, "Loading models...");
  assert.equal(startupMessageFromLog(models, "unrelated output"), models);

  const server = startupMessageFromLog(models, "  - Starting server...");
  assert.equal(server, "Starting server...");
  assert.equal(
    startupMessageFromLog(
      server,
      "  - loading PyTorch, Unsloth and Transformers...",
    ),
    server,
  );
});

test("server fallback delay is three seconds", () => {
  assert.equal(SERVER_START_FALLBACK_MS, 3_000);
});
