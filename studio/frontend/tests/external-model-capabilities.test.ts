// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The policy an external connection's selection applies to the runtime store.
//
// It lived inline in chat-page.tsx's selection effect, out of the compare panes' reach, so the
// compare path carried a copy -- and the copy drifted: it kept calling the effort clamp that
// collapses `max`/`xhigh` onto `low`, and hard-coded Unsloth tools off. Both surfaces now call
// one function, and the last test here pins them to the same clamp at the source level, which
// is the only reachable check for the part the node runner cannot import.

import assert from "node:assert/strict";
import test from "node:test";

import { deriveExternalModelCapabilities } from "../src/features/chat/lib/external-model-capabilities.ts";

import { readSrc } from "./helpers/kit.ts";

const CAPS = {
  supportsReasoning: true,
  reasoningAlwaysOn: false,
  reasoningStyle: "reasoning_effort" as const,
  supportsReasoningOff: true,
  reasoningEffortLevels: ["low", "medium", "high"],
  defaultEffort: null,
};

const BASE = {
  providerType: "openrouter",
  reasoningCaps: CAPS,
  clampedCurrentEffort: "medium" as const,
  currentReasoningEffort: "medium" as const,
  currentReasoningEnabled: true,
  supportsBuiltinWebSearch: false,
  supportsBuiltinCodeExecution: false,
  supportsBuiltinImageGeneration: false,
  supportsBuiltinWebFetch: false,
  supportsStudioTools: false,
  providerHostsCodeExecution: false,
  storedToolsEnabled: null,
  storedCodeToolsEnabled: null,
  storedImageToolsEnabled: null,
  storedWebFetchToolsEnabled: null,
};

// Anthropic and OpenAI both return structured citations, so search comes up on for them alone.
test("search defaults on for Anthropic and OpenAI, off for everyone else", () => {
  for (const providerType of ["anthropic", "openai"]) {
    const caps = deriveExternalModelCapabilities({
      ...BASE,
      providerType,
      supportsBuiltinWebSearch: true,
    });
    assert.equal(caps.toolsEnabled, true, providerType);
  }
  const openrouter = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "openrouter",
    supportsBuiltinWebSearch: true,
  });
  assert.equal(openrouter.toolsEnabled, false);
});

test("a stored pill outranks the per-provider default, in both directions", () => {
  const off = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "anthropic",
    supportsBuiltinWebSearch: true,
    storedToolsEnabled: false,
  });
  assert.equal(off.toolsEnabled, false);
  const on = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "openrouter",
    supportsBuiltinWebSearch: true,
    storedToolsEnabled: true,
  });
  assert.equal(on.toolsEnabled, true);
});

// Kimi's k2.6/k2.5 default to thinking enabled server-side, so Think comes up clicked and Search
// stays off; the composer's mutual-exclusion handlers flip the two.
test("Kimi comes up thinking with search off, whatever was stored", () => {
  const caps = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "kimi",
    supportsBuiltinWebSearch: true,
    storedToolsEnabled: true,
    currentReasoningEnabled: false,
  });
  assert.equal(caps.toolsEnabled, false);
  assert.equal(caps.reasoningEnabled, true);
});

test("Anthropic takes the top rung offered, OpenAI takes high", () => {
  const anthropic = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "anthropic",
    reasoningCaps: {
      ...CAPS,
      reasoningEffortLevels: ["low", "medium", "high", "xhigh"],
    },
  });
  assert.equal(anthropic.reasoningEffort, "xhigh");
  const openai = deriveExternalModelCapabilities({ ...BASE, providerType: "openai" });
  assert.equal(openai.reasoningEffort, "high");
});

// The catalog's own default is the model's, so it outranks the per-provider rule of thumb.
test("a catalog default effort outranks the per-provider default", () => {
  const caps = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "anthropic",
    reasoningCaps: { ...CAPS, defaultEffort: "low" },
  });
  assert.equal(caps.reasoningEffort, "low");
});

test("a catalog default the model does not offer is ignored", () => {
  const caps = deriveExternalModelCapabilities({
    ...BASE,
    providerType: "openai",
    reasoningCaps: { ...CAPS, defaultEffort: "xhigh" },
  });
  assert.equal(caps.reasoningEffort, "high");
});

test("a model that cannot reason keeps the effort the user already had", () => {
  const caps = deriveExternalModelCapabilities({
    ...BASE,
    currentReasoningEffort: "low",
    clampedCurrentEffort: "medium",
    reasoningCaps: { ...CAPS, supportsReasoning: false },
  });
  assert.equal(caps.reasoningEffort, "low");
  assert.equal(caps.supportsReasoning, false);
});

// The regression the compare copy shipped: Unsloth tools were hard-coded off, so a self-hosted
// connection that CAN run them through the loop silently could not.
test("a provider that runs Unsloth tools reports them supported and can search", () => {
  const caps = deriveExternalModelCapabilities({
    ...BASE,
    supportsStudioTools: true,
    storedToolsEnabled: true,
  });
  assert.equal(caps.supportsTools, true);
  assert.equal(caps.toolsEnabled, true);
});

// The other half of that regression: code ran off the raw builtin flag rather than the
// placement rule, so a sandbox-owning provider that cannot use it still looked runnable.
test("code follows the placement rule, not the hosted flag alone", () => {
  const cannotRun = deriveExternalModelCapabilities({
    ...BASE,
    supportsBuiltinCodeExecution: false,
    providerHostsCodeExecution: true,
    supportsStudioTools: false,
    storedCodeToolsEnabled: true,
  });
  assert.equal(cannotRun.codeToolsEnabled, false);
});

test("preserve-thinking is always cleared for an external model", () => {
  for (const providerType of ["anthropic", "openai", "kimi", "openrouter", "custom"]) {
    const caps = deriveExternalModelCapabilities({ ...BASE, providerType });
    assert.equal(caps.supportsPreserveThinking, false, providerType);
  }
});

test("an unsupported builtin forces its pill off even when one was stored on", () => {
  const caps = deriveExternalModelCapabilities({
    ...BASE,
    storedImageToolsEnabled: true,
    storedWebFetchToolsEnabled: true,
  });
  assert.equal(caps.imageToolsEnabled, false);
  assert.equal(caps.webFetchToolsEnabled, false);
});

// The historical drift, pinned where the node runner cannot reach: provider-capabilities.ts
// imports api/providers-api.ts, which pulls node-forge and a `@/` alias, so neither call site
// can be imported here. clampLocalReasoningEffort collapses `max`/`xhigh` onto `low`; the
// levels-aware clamp is the one both surfaces must feed this function.
test("both call sites clamp with the levels-aware helper", () => {
  for (const file of ["features/chat/chat-page.tsx", "features/chat/shared-composer.tsx"]) {
    const src = readSrc(file);
    if (!src.includes("deriveExternalModelCapabilities")) continue;
    assert.match(
      src,
      /clampedCurrentEffort:\s*clampReasoningEffortToLevels\(/,
      `${file} must pass the levels-aware clamp`,
    );
  }
});
