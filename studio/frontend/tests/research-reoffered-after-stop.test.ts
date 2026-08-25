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

test("a finished research reply still holds the thread", () => {
  for (const status of [
    "completed",
    "running",
    "awaiting_approval",
    "failed",
  ]) {
    assert.equal(
      messageHasResearchRunId(
        message({ researchRunId: "run_1", researchStatus: status }),
      ),
      true,
      status,
    );
  }
  assert.equal(
    messageHasResearchRunId(message({ researchRunId: "run_1" })),
    true,
  );
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
