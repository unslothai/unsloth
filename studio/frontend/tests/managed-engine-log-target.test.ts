// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { managedEngineFailureLogTarget } from "../src/features/settings/lib/managed-engine-log-target.ts";

test("each managed engine failure keeps its own log target", () => {
  const path = "C:\\Unsloth Home\\logs\\sglang\\sglang-123.log";
  const detail = {
    code: "managed_engine_load_failed",
    log_family: "sglang",
    log_path: path,
  };
  const first = managedEngineFailureLogTarget(detail);
  const second = managedEngineFailureLogTarget({
    ...detail,
    log_path: "/tmp/logs/sglang/sglang-456.log",
  });
  assert.deepEqual(first, { family: "sglang", path });
  assert.notEqual(first?.path, second?.path);
});

test("log targets require a structured managed launch failure and a path", () => {
  assert.equal(managedEngineFailureLogTarget("Full log: /tmp/sglang.log"), null);
  assert.equal(managedEngineFailureLogTarget({ code: "managed_engine_load_failed", log_family: "server", log_path: "/tmp/server.log" }), null);
  assert.equal(managedEngineFailureLogTarget({ code: "managed_engine_load_failed", log_family: "sglang", log_path: "" }), null);
});

test("vLLM failures select the vLLM family and keep the launch path", () => {
  const log = {
    code: "managed_engine_load_failed",
    log_family: "vllm",
    log_path: "/tmp/logs/vllm/vllm-123.log",
  };
  assert.deepEqual(managedEngineFailureLogTarget(log), {
    family: "vllm",
    path: log.log_path,
  });
});
