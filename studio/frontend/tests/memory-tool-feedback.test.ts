// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";
import { memoryToolFeedback } from "../src/components/assistant-ui/memory-tool-feedback.ts";

const saved = {
  phase: "settled",
  forgetting: false,
  project: false,
  resultText: "Memory updated: personal.note_7afc",
} as const;

test("memory feedback describes the outcome without exposing storage keys", () => {
  assert.deepEqual(memoryToolFeedback(saved), {
    label: "Saved for future chats",
    detail: "",
    canManage: true,
  });
  assert.equal(
    memoryToolFeedback({ ...saved, project: true }).label,
    "Saved for this project",
  );
  assert.equal(
    memoryToolFeedback({ ...saved, forgetting: true }).label,
    "Forgotten",
  );
  const unchanged = {
    ...saved,
    resultText:
      "Memory unchanged: personal.note_7afc. Requested state already holds; answer the user without repeating this call.",
  };
  assert.equal(memoryToolFeedback(unchanged).label, "Already remembered");
  assert.equal(
    memoryToolFeedback({ ...unchanged, forgetting: true }).label,
    "Already forgotten",
  );
});

test("pending and cancelled updates do not claim success", () => {
  for (const phase of ["approval", "running", "cancelled"] as const) {
    const feedback = memoryToolFeedback({ ...saved, phase });
    assert.equal(feedback.canManage, false);
    assert.doesNotMatch(feedback.label, /Saved|Forgotten/);
  }
});

test("failures explain recovery without leaking model validation instructions", () => {
  const full = memoryToolFeedback({
    ...saved,
    resultText: "Error: Memory is full. Shorten or remove an existing memory.",
  });
  assert.equal(full.canManage, true);
  assert.match(full.detail, /Remove a saved memory/);
  const rejected = memoryToolFeedback({
    ...saved,
    resultText: "Error: Use a short lowercase key with underscores.",
  });
  assert.equal(rejected.detail, "Nothing was changed. Try again.");
  assert.doesNotMatch(JSON.stringify(rejected), /key|underscores/);
  assert.equal(
    memoryToolFeedback({
      ...saved,
      forgetting: true,
      resultText: "Error: Supply value, or null to forget a fact.",
    }).label,
    "Couldn't forget this memory",
  );
  const unavailable = memoryToolFeedback({
    ...saved,
    resultText: "Error: Choose personal or the current project memory.",
  });
  assert.match(unavailable.detail, /Open a chat/);
  assert.doesNotMatch(unavailable.detail, /scope|personal/);
});
