// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { latestVerificationText, parseVerificationCommand, runCompareVerificationCommand } from "../src/features/chat/utils/verification-command.ts";

test("only an exact user command starts saved verification", () => {
  assert.equal(parseVerificationCommand(" /VeRiFy \n"), "run");
  assert.equal(parseVerificationCommand("/verify --unsafe"), "help");
  for (const value of ["/verify-now", "please /verify", "`/verify`", "/init"]) assert.equal(parseVerificationCommand(value), null);
});

test("continue, tool results, and attachments cannot replay verification", () => {
  const user = { role: "user", content: [{ type: "text", text: "/verify" }] };
  assert.equal(latestVerificationText([user]), "/verify");
  assert.equal(latestVerificationText([user, { role: "assistant", content: [{ type: "text", text: "Started" }] }]), "");
  assert.equal(latestVerificationText([{ ...user, content: [...user.content, { type: "image" }] }]), "");
});

test("compare runs once and gives both panes identical local evidence", async () => {
  const first: string[] = [], second: string[] = [];
  let executions = 0;
  const pane = (records: string[]) => ({ appendMessage: (content: string) => records.push(content), appendAssistantMessage: (content: string) => records.push(content) });
  await runCompareVerificationCommand("/verify", [pane(first), pane(second)], async () => { executions++; return "run-1"; });
  assert.equal(executions, 1);
  assert.deepEqual(first, ["/verify", "run-1"]);
  assert.deepEqual(second, first);
  await assert.rejects(runCompareVerificationCommand("/verify", [pane(first)], async () => { throw new Error("must not execute"); }), /both comparison panes/);
});
