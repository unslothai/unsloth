// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend re-points a stopped run at the next question instead of refusing it, so the
// composer has to offer deep research again. If either half of this drifts the user is locked
// out of research for a chat the server would happily research.

import assert from "node:assert/strict";
import test from "node:test";

import {
  messageHasResearchRunId,
  threadHasResearchMessage,
} from "../src/components/assistant-ui/thread-research-presence.ts";

const message = (custom: Record<string, unknown>) => ({ metadata: { custom } });

test("a finished research reply holds the thread", () => {
  for (const status of ["completed", "failed"]) {
    assert.equal(
      messageHasResearchRunId(
        message({ researchRunId: "run_1", researchStatus: status }),
      ),
      true,
      status,
    );
  }
  // No status at all is older metadata: count it, to stay on the safe side.
  assert.equal(
    messageHasResearchRunId(message({ researchRunId: "run_1" })),
    true,
  );
});

test("a reply whose run is still going keeps the toggle lit instead", () => {
  for (const status of ["planning", "queued", "running", "cancelling"]) {
    assert.equal(
      messageHasResearchRunId(
        message({ researchRunId: "run_1", researchStatus: status }),
      ),
      false,
      status,
    );
  }
});

test("a stopped research reply does not, from history or from a live run", () => {
  assert.equal(
    messageHasResearchRunId(
      message({ researchRunId: "run_1", researchStatus: "cancelled" }),
    ),
    false,
  );
  assert.equal(
    messageHasResearchRunId(
      message({ researchRunId: "run_1", researchRun: { status: "cancelled" } }),
    ),
    false,
  );
});

test("the live run wins over the stored status, which lags a stop by one write", () => {
  assert.equal(
    messageHasResearchRunId(
      message({
        researchRunId: "run_1",
        researchStatus: "running",
        researchRun: { status: "cancelled" },
      }),
    ),
    false,
  );
});

test("a live retry overrides the failed status stored on the assistant message", () => {
  const messages = [
    message({ researchRunId: "run_1", researchStatus: "failed" }),
  ];

  assert.equal(threadHasResearchMessage(messages), true);
  assert.equal(threadHasResearchMessage(messages, "run_1"), false);
});

test("a thread whose only research reply was stopped reads as unused", () => {
  assert.equal(
    threadHasResearchMessage([
      {},
      message({ researchRunId: "run_1", researchStatus: "cancelled" }),
    ]),
    false,
  );
  assert.equal(
    threadHasResearchMessage([
      message({ researchRunId: "run_1", researchStatus: "cancelled" }),
      message({ researchRunId: "run_2", researchStatus: "completed" }),
    ]),
    true,
  );
});

test("a message with no run id is unaffected", () => {
  assert.equal(messageHasResearchRunId({}), false);
  assert.equal(messageHasResearchRunId(message({})), false);
  assert.equal(
    messageHasResearchRunId(message({ researchStatus: "cancelled" })),
    false,
  );
});
