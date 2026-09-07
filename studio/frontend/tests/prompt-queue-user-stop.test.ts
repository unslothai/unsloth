import assert from "node:assert/strict";
import test from "node:test";

import {
  planUserPromptQueueStop,
  userStopTargetCancelMode,
} from "../src/features/chat/utils/prompt-queue-user-stop.ts";

class SharedQueueTarget {
  permanentlyCancelled = false;
  activeRunCancels = 0;
  cancel() {
    this.permanentlyCancelled = true;
  }
  cancelActiveRun() {
    this.activeRunCancels += 1;
  }
  append() {
    return this.permanentlyCancelled ? "skipped" : "sent";
  }
}

function applyStopCancel(
  target: SharedQueueTarget,
  plan: ReturnType<typeof planUserPromptQueueStop>,
) {
  const mode = userStopTargetCancelMode(plan);
  if (mode === "permanent") target.cancel();
  else if (mode === "activeRun") target.cancelActiveRun();
  return mode;
}

const plans: Array<{
  name: string;
  items: Array<{ dispatched: boolean }>;
  index: number;
  want: ReturnType<typeof planUserPromptQueueStop>;
}> = [
  {
    name: "a dispatched stop keeps pending follow-ups and pauses",
    items: [{ dispatched: true }, { dispatched: false }, { dispatched: false }],
    index: 0,
    want: { cancelActiveItem: true, retainedItemIndexes: [1, 2], pause: true },
  },
  {
    name: "a dispatched stop with no follow-ups leaves nothing to resume",
    items: [{ dispatched: true }],
    index: 0,
    want: { cancelActiveItem: true, retainedItemIndexes: [], pause: false },
  },
  {
    name: "an undispatched stop keeps the whole queue and pauses",
    items: [{ dispatched: false }, { dispatched: false }],
    index: 0,
    want: { cancelActiveItem: false, retainedItemIndexes: [0, 1], pause: true },
  },
  {
    name: "a queue still waiting to start is paused without cancelling",
    items: [{ dispatched: false }, { dispatched: false }],
    index: -1,
    want: { cancelActiveItem: false, retainedItemIndexes: [0, 1], pause: true },
  },
];

for (const entry of plans) {
  test(entry.name, () => {
    assert.deepEqual(planUserPromptQueueStop(entry.items, entry.index), entry.want);
  });
}

test("pausing a shared-target batch still lets the next prompt append", () => {
  const target = new SharedQueueTarget();
  const plan = planUserPromptQueueStop(
    [{ dispatched: true }, { dispatched: false }],
    0,
  );
  assert.equal(applyStopCancel(target, plan), "activeRun");
  assert.equal(target.append(), "sent");
  assert.equal(target.activeRunCancels, 1);
  assert.equal(target.permanentlyCancelled, false);
});

test("a stop with nothing left permanently cancels the shared target", () => {
  const target = new SharedQueueTarget();
  const plan = planUserPromptQueueStop([{ dispatched: true }], 0);
  assert.equal(applyStopCancel(target, plan), "permanent");
  assert.equal(target.append(), "skipped");
});
