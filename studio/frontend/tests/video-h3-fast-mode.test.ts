// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// MiniMax-H3 4-step fast mode: opt-in, sent only on an H3 fl2va load, unavailable with a reason when the adapter is.

import assert from "node:assert/strict";
import test from "node:test";

import {
  H3_FAST_DRAFT_NOTE,
  H3_FAST_LABEL,
  H3_FAST_STEPS,
  h3FastEngaged,
  h3FastLoadFields,
  h3FastNeedsReload,
  h3FastToggleState,
} from "../src/features/video/h3-fast-mode.ts";
import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/video/video-page.tsx");
const api = readSrc("features/video/api.ts");

test("the load field is sent only when on, for an H3 fl2va pick", () => {
  assert.deepEqual(h3FastLoadFields(false, "pipeline", "fl2va"), {});
  assert.deepEqual(h3FastLoadFields(true, "pipeline", undefined), { h3_fast_4step: true });
  assert.deepEqual(h3FastLoadFields(true, "pipeline", "fl2va"), { h3_fast_4step: true });
  assert.deepEqual(h3FastLoadFields(true, "single_file", "fl2va"), { h3_fast_4step: true });
  assert.deepEqual(h3FastLoadFields(true, "pipeline", "ref2va"), {});
  assert.deepEqual(h3FastLoadFields(true, "gguf", "fl2va"), {});
});

test("the switch shows for H3 only and says why it is off", () => {
  const ok = { available: true, repo: "unsloth/pi-H3" };
  assert.equal(h3FastToggleState({ isH3: false, kind: "pipeline", h3Task: "fl2va", availability: ok }).visible, false);
  assert.deepEqual(h3FastToggleState({ isH3: true, kind: "pipeline", h3Task: "fl2va", availability: ok }), {
    visible: true,
    disabled: false,
    reason: null,
  });
  const missing = { available: false, reason: "The 4-step fast mode is unavailable: unsloth/pi-H3 is not available." };
  const off = h3FastToggleState({ isH3: true, kind: "pipeline", h3Task: "fl2va", availability: missing });
  assert.equal(off.disabled, true);
  assert.match(off.reason ?? "", /unsloth\/pi-H3/);
  assert.match(h3FastToggleState({ isH3: true, kind: "pipeline", h3Task: "ref2va", availability: ok }).reason ?? "", /Reference-to-video/);
  assert.equal(h3FastToggleState({ isH3: true, kind: "gguf", h3Task: "fl2va", availability: ok }).disabled, true);
  assert.equal(h3FastToggleState({ isH3: true, kind: "pipeline", h3Task: "fl2va", availability: null }).disabled, true);
});

test("engaged and reload state follow the resident build", () => {
  const on = { loaded: true, family: "minimax-h3", h3_fast: { enabled: true, steps: 4, guidance: 1 } };
  const off = { loaded: true, family: "minimax-h3", h3_fast: null };
  assert.equal(H3_FAST_STEPS, 4);
  assert.equal(h3FastEngaged(on), true);
  assert.equal(h3FastEngaged(off), false);
  assert.equal(h3FastNeedsReload(true, off), true);
  assert.equal(h3FastNeedsReload(false, on), true);
  assert.equal(h3FastNeedsReload(true, on), false);
  assert.equal(h3FastNeedsReload(true, { loaded: true, family: "wan2.2" }), false);
});

test("the page wires the switch into the load and the advanced panel", () => {
  // off by default, carried through the load controls, sent through the helper only
  assert.match(page, /const \[h3Fast, setH3Fast\] = useState\(false\);/);
  assert.match(page, /h3_fast_4step: controls\.h3Fast \? true : undefined,/);
  assert.match(page, /h3FastLoadFields\(advanced\.h3_fast_4step === true, opts\.kind, opts\.h3Task\)/);
  assert.equal(page.split("h3_fast_4step: true").length - 1, 0, "the page never hard-codes the field");
  assert.match(page, /aria-label=\{H3_FAST_LABEL\}/);
  assert.match(page, /getVideoH3FastAvailability\(ctl\.signal\)/);
  assert.match(api, /"\/api\/inference\/video\/h3-fast-4step"/);
  assert.match(api, /h3_fast\?: H3FastStatus \| null;/);
});

test("the mode is labelled a draft, with the prompt-adherence caveat", () => {
  assert.match(H3_FAST_LABEL, /draft/);
  assert.match(H3_FAST_DRAFT_NOTE, /prompt adherence can drop on some prompts/);
  assert.match(page, /\{H3_FAST_DRAFT_NOTE\}/);
  // the loaded build names it, and an automatic reload that dropped it says why
  assert.match(page, /status\.h3_fast\?\.enabled \? \(\s*<BuildRow label="Steps" value=\{`\$\{status\.h3_fast\.steps\} · \$\{H3_FAST_LABEL\}`\} \/>/);
  assert.match(page, /status\?\.h3_fast_dropped \?/);
  assert.match(api, /h3_fast_dropped\?: string \| null;/);
});

test("the download plan asks for the adapter whenever the load will", () => {
  const plan = page.slice(page.indexOf("const plan = await getVideoDownloadPlan({"), page.indexOf("// Superseded; report started"));
  assert.match(plan, /h3FastLoadFields\(advanced\.h3_fast_4step === true, opts\.kind, opts\.h3Task\)/);
  assert.equal(page.split("h3FastLoadFields(advanced.h3_fast_4step === true, opts.kind, opts.h3Task)").length - 1, 2);
});
