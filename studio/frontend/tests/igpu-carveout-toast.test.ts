// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every model load calls showCarveoutAdvice, including the loads carrying no advice,
// so this is where one model's numbers could outlive the model they describe, two
// loads could stack two notices over the composer, or a dismissal could report an
// allocation the user is no longer running.
//
// One resolver redirects both dependencies: the toast stub records the full options
// bag (id, duration, action -- none of which the store-stubs toast keeps), and the
// auth stub answers authFetch so no test reaches the network.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/igpu-carveout-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const {
  IGPU_CARVEOUT_ACTION_CLASS,
  IGPU_CARVEOUT_NOTICE_TITLE,
  IGPU_CARVEOUT_TOAST_ID,
  dismissCarveoutAdviceForModel,
  showCarveoutAdvice,
} = await import("../src/features/igpu-carveout/igpu-carveout-toast.ts");

const ADVICE = {
  current_gb: 32,
  needed_gb: 42.9,
  suggested_gb: 48,
  machine_gb: 127.8,
  host_left_gb: 79.8,
  message:
    "Weights need about 43 GB but only 32 GB is allocated to the integrated GPU, " +
    "so the rest runs from slower shared memory. Raising it to 48 GB in your " +
    "firmware or GPU control panel leaves about 80 GB for the system.",
};

function reset() {
  calls.length = 0;
  setAuthFetchHandler(null);
}

test("advice raises one toast carrying the backend's prose", () => {
  reset();
  showCarveoutAdvice(ADVICE);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].kind, "info");
  assert.equal(calls[0].title, IGPU_CARVEOUT_NOTICE_TITLE);
  assert.equal(calls[0].options?.description, ADVICE.message);
  // Never an error or a warning: the load succeeded and the user did nothing wrong.
  assert.notEqual(calls[0].kind, "error");
});

test("it is not a modal, so it holds an id and a finite duration", () => {
  // A toast without an id stacks, and one without a duration stays until it is
  // clicked, which is a dialog with extra steps.
  reset();
  showCarveoutAdvice(ADVICE);
  assert.equal(calls[0].options?.id, IGPU_CARVEOUT_TOAST_ID);
  assert.ok(
    Number.isFinite(calls[0].options?.duration) &&
      (calls[0].options?.duration as number) > 5000,
    "long enough to read an action, short enough to leave on its own",
  );
});

test("the action is styled down from sonner's filled default", () => {
  // An outline because nothing here is the thing to do, and end-alignment because the
  // shared toast CSS starts an action at the left and floats it under six lines.
  reset();
  showCarveoutAdvice(ADVICE);
  const classes = (calls[0].options?.classNames as { actionButton?: string })
    ?.actionButton;
  assert.equal(classes, IGPU_CARVEOUT_ACTION_CLASS);
  assert.match(classes ?? "", /!justify-self-end/);
  assert.match(classes ?? "", /!bg-transparent/);
});

test("a second load replaces the notice rather than stacking one over it", () => {
  // Two toasts over the composer, the older one stale, is what the id prevents.
  reset();
  showCarveoutAdvice(ADVICE);
  showCarveoutAdvice({ ...ADVICE, suggested_gb: 96 });
  assert.equal(calls.length, 2);
  assert.equal(calls[0].options?.id, calls[1].options?.id);
});

test("a malformed payload raises nothing", () => {
  reset();
  for (const bad of [null, undefined, {}, "advice", { ...ADVICE, message: "" }]) {
    showCarveoutAdvice(bad);
  }
  assert.equal(
    calls.filter((call) => call.kind !== "dismiss").length,
    0,
    "the notice quotes numbers, so a partial payload must produce no toast",
  );
});

test("a later load carrying no advice takes the previous notice down", () => {
  // The moment the old figures stop being true; a toast has no copy, so it dismisses.
  reset();
  showCarveoutAdvice(ADVICE);
  showCarveoutAdvice(undefined);
  const dismissals = calls.filter((call) => call.kind === "dismiss");
  assert.equal(dismissals.length, 1);
  assert.equal(dismissals[0].id, IGPU_CARVEOUT_TOAST_ID);
});

test("the action posts the allocation it was dismissed at", async () => {
  reset();
  const seen: Array<{ url: string; body: unknown; method?: string }> = [];
  setAuthFetchHandler((url, init) => {
    seen.push({
      url,
      method: init?.method,
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    return new Response(JSON.stringify({ dismissed_at_gb: 32 }), { status: 200 });
  });
  showCarveoutAdvice(ADVICE);
  const action = calls[0].options?.action as { onClick: () => void };
  action.onClick();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(seen.length, 1);
  assert.equal(seen[0].method, "POST");
  assert.match(seen[0].url, /igpu-carveout-notice\/dismiss/);
  assert.deepEqual(seen[0].body, { current_gb: 32 });
});

test("letting the toast expire sends nothing", async () => {
  // Ignoring a toast is not a decision, so the notice may return on a later load.
  reset();
  let called = false;
  setAuthFetchHandler(() => {
    called = true;
    return new Response("{}", { status: 200 });
  });
  showCarveoutAdvice(ADVICE);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(called, false);
});

test("a failing dismissal is swallowed rather than rejected", async () => {
  // Worst case the notice returns on a later load, which beats an unhandled
  // rejection.
  reset();
  setAuthFetchHandler(() => {
    throw new TypeError("network down");
  });
  showCarveoutAdvice(ADVICE);
  const action = calls[0].options?.action as { onClick: () => void };
  assert.doesNotThrow(() => action.onClick());
  await new Promise((resolve) => setTimeout(resolve, 0));
});

test("unloading the advised model takes its notice down", () => {
  // Unload inside the toast's 12 seconds and "this model could run faster" is false,
  // with no load coming to correct it.
  reset();
  showCarveoutAdvice(ADVICE, "/models/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("/models/qwen3-30b.gguf");
  const dismissed = calls.filter((call) => call.kind === "dismiss");
  assert.equal(dismissed.length, 1);
  assert.equal(dismissed[0].id, IGPU_CARVEOUT_TOAST_ID);
});

test("unloading a different model leaves it up", () => {
  // Several models can be resident at once, and the notice is about one of them.
  reset();
  showCarveoutAdvice(ADVICE, "/models/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("/models/gemma3-27b.gguf");
  assert.equal(
    calls.filter((call) => call.kind === "dismiss").length,
    0,
  );
});

test("every dismissal names this notice and only this notice", () => {
  // The id also keeps these dismissals off every other toast on screen.
  reset();
  showCarveoutAdvice(ADVICE, "/models/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("/models/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("/models/gemma3-27b.gguf");
  showCarveoutAdvice(null);
  for (const call of calls.filter((entry) => entry.kind === "dismiss")) {
    assert.equal(call.id, IGPU_CARVEOUT_TOAST_ID);
  }
});

test("either identity the load was known by takes the notice down", () => {
  // A cached Hub candidate is requested by its loadId while the unload uses the
  // checkpoint. Matching one identity only left the toast up for a gone model.
  reset();
  showCarveoutAdvice(ADVICE, "unsloth/Qwen3-30B-GGUF", "/cache/hub/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("unsloth/Qwen3-30B-GGUF");
  assert.equal(calls.filter((call) => call.kind === "dismiss").length, 1);

  reset();
  showCarveoutAdvice(ADVICE, "unsloth/Qwen3-30B-GGUF", "/cache/hub/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("/cache/hub/qwen3-30b.gguf");
  assert.equal(calls.filter((call) => call.kind === "dismiss").length, 1);

  // And a third model is still not this one.
  reset();
  showCarveoutAdvice(ADVICE, "unsloth/Qwen3-30B-GGUF", "/cache/hub/qwen3-30b.gguf");
  dismissCarveoutAdviceForModel("unsloth/gemma-3-27b");
  assert.equal(calls.filter((call) => call.kind === "dismiss").length, 0);
});
