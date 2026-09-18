// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What an external connection's selection does to the runtime store, in one place for every
// surface that selects one.
//
// This was inline in chat-page.tsx's selection effect, so the compare panes could not reach it
// and would have needed a second copy. A second copy is how the compare path drifted before: it
// went on calling the effort clamp that collapses `max`/`xhigh` onto `low` long after the
// single-model path had moved to the levels-aware one, and it hard-coded Unsloth tools off.
//
// The POLICY lives here -- per-provider effort defaults, the Kimi rules, search-on-by-default,
// how a stored pill outranks a default, and clearing preserve-thinking. That is the part that
// drifted. The capability LOOKUPS stay with the callers and arrive as data, because
// provider-capabilities.ts reaches api/providers-api.ts, which pulls node-forge and a `@/` alias
// the node runner cannot resolve -- importing it here would make this module untestable, which
// is the thing being fixed. `tests/external-model-capabilities.test.ts` additionally pins both
// call sites to the levels-aware clamp at the source level, so the original drift cannot recur.

import { codeToolCanRun } from "../api/code-tool-placement.ts";
import type { ExternalReasoningCapabilities } from "../provider-capabilities";
import type {
  ReasoningEffort,
  ReasoningStyle,
} from "../stores/chat-runtime-store";

export type ExternalModelCapabilityInput = {
  /** Drives the per-provider defaults only; the capability booleans are looked up by the caller. */
  providerType: string | null | undefined;
  reasoningCaps: ExternalReasoningCapabilities;
  /** `clampReasoningEffortToLevels(currentEffort, reasoningCaps.reasoningEffortLevels)`. */
  clampedCurrentEffort: ReasoningEffort;
  /** The effort in the store now; kept as-is when the model cannot reason. */
  currentReasoningEffort: ReasoningEffort;
  /** Whether thinking is on now; kept where the model lets it be turned off. */
  currentReasoningEnabled: boolean;
  supportsBuiltinWebSearch: boolean;
  supportsBuiltinCodeExecution: boolean;
  supportsBuiltinImageGeneration: boolean;
  supportsBuiltinWebFetch: boolean;
  /** This provider AND model can run Unsloth's own tools through the loop. */
  supportsStudioTools: boolean;
  providerHostsCodeExecution: boolean;
  // null = nothing stored, so the per-provider default decides.
  storedToolsEnabled: boolean | null;
  storedCodeToolsEnabled: boolean | null;
  storedImageToolsEnabled: boolean | null;
  storedWebFetchToolsEnabled: boolean | null;
};

export type ExternalModelCapabilityState = {
  supportsReasoning: boolean;
  reasoningAlwaysOn: boolean;
  reasoningStyle: ReasoningStyle;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: ExternalReasoningCapabilities["reasoningEffortLevels"];
  reasoningEffort: ReasoningEffort;
  reasoningEnabled: boolean;
  supportsPreserveThinking: boolean;
  supportsTools: boolean;
  supportsBuiltinWebSearch: boolean;
  supportsBuiltinCodeExecution: boolean;
  supportsBuiltinImageGeneration: boolean;
  supportsBuiltinWebFetch: boolean;
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
  imageToolsEnabled: boolean;
  webFetchToolsEnabled: boolean;
};

/** The effort an external selection comes up at.
 *
 *  The model's own catalog default wins where it offers one. Otherwise Anthropic takes the top
 *  rung, since Claude's adaptive thinking adjusts cost per turn; OpenAI takes "high"; everyone
 *  else "medium". All of it is overridable via Think. A model that cannot reason keeps whatever
 *  the user already had rather than being moved. */
function resolveDefaultEffort(
  providerType: string | null | undefined,
  reasoningCaps: ExternalReasoningCapabilities,
  clampedCurrentEffort: ReasoningEffort,
  currentReasoningEffort: ReasoningEffort,
): ReasoningEffort {
  if (!reasoningCaps.supportsReasoning) return currentReasoningEffort;
  const effortLevels = reasoningCaps.reasoningEffortLevels;
  const offers = (level: ReasoningEffort) => effortLevels.includes(level);
  if (reasoningCaps.defaultEffort && offers(reasoningCaps.defaultEffort)) {
    return reasoningCaps.defaultEffort;
  }
  if (providerType === "anthropic") {
    if (offers("xhigh")) return "xhigh";
    return offers("high") ? "high" : clampedCurrentEffort;
  }
  if (providerType === "openai") {
    if (offers("high")) return "high";
    return offers("medium") ? "medium" : clampedCurrentEffort;
  }
  return offers("medium") ? "medium" : clampedCurrentEffort;
}

/** Whether thinking comes up on.
 *
 *  A model that cannot turn thinking off is always on. Kimi's k2.6/k2.5 default to thinking
 *  enabled server-side, so its pill comes up clicked; everyone else keeps what the chat had. */
function resolveReasoningEnabled(
  providerType: string | null | undefined,
  reasoningCaps: ExternalReasoningCapabilities,
  currentReasoningEnabled: boolean,
): boolean {
  if (!reasoningCaps.supportsReasoning) return currentReasoningEnabled;
  if (!reasoningCaps.supportsReasoningOff) return true;
  return providerType === "kimi" ? true : currentReasoningEnabled;
}

export function deriveExternalModelCapabilities({
  providerType,
  reasoningCaps,
  clampedCurrentEffort,
  currentReasoningEffort,
  currentReasoningEnabled,
  supportsBuiltinWebSearch,
  supportsBuiltinCodeExecution,
  supportsBuiltinImageGeneration,
  supportsBuiltinWebFetch,
  supportsStudioTools,
  providerHostsCodeExecution,
  storedToolsEnabled,
  storedCodeToolsEnabled,
  storedImageToolsEnabled,
  storedWebFetchToolsEnabled,
}: ExternalModelCapabilityInput): ExternalModelCapabilityState {
  const effortLevels = reasoningCaps.reasoningEffortLevels;
  const isAnthropic = providerType === "anthropic";
  const isOpenAI = providerType === "openai";
  const nextReasoningEffort = resolveDefaultEffort(
    providerType,
    reasoningCaps,
    clampedCurrentEffort,
    currentReasoningEffort,
  );
  // Kimi's k2.6/k2.5 default to thinking enabled server-side, so the Think pill comes up clicked. Search stays
  // off; the composer's mutual-exclusion handlers flip the two. Per https://platform.kimi.ai/docs/models.
  const isKimi = providerType === "kimi";
  // Web search on by default only for Anthropic and OpenAI, both with structured citations.
  // OpenRouter and Kimi work on opt-in but are less reliable.
  const searchOnByDefault =
    supportsBuiltinWebSearch && (isAnthropic || isOpenAI);
  // Unsloth runs Search and Code itself for any provider that advertises the capability, so a self-hosted
  // connection has no hosted builtin to key off. Keying the pill state on the hosted flags alone discarded the
  // saved preference on every reload and sent enable_tools: false.
  const canSearch = supportsBuiltinWebSearch || supportsStudioTools;
  // Read out of the placement rule, not off the Unsloth-tools flag: a model on a sandbox-owning
  // provider that cannot use it runs nothing either way.
  const canRunCode = codeToolCanRun({
    hostedCodeExecutionForThisTurn: supportsBuiltinCodeExecution,
    providerHostsCodeExecution,
    supportsStudioTools,
  });
  // Kimi flips Think and Search as a pair, so Search never comes up on for it.
  const nextToolsEnabled =
    canSearch && !isKimi ? (storedToolsEnabled ?? searchOnByDefault) : false;
  return {
    supportsReasoning: reasoningCaps.supportsReasoning,
    reasoningAlwaysOn: reasoningCaps.reasoningAlwaysOn,
    reasoningStyle: reasoningCaps.reasoningStyle,
    supportsReasoningOff: reasoningCaps.supportsReasoningOff,
    reasoningEffortLevels: effortLevels,
    reasoningEffort: nextReasoningEffort,
    reasoningEnabled: resolveReasoningEnabled(
      providerType,
      reasoningCaps,
      currentReasoningEnabled,
    ),
    // A local-runtime concept, so an external selection clears it rather than inheriting
    // whatever the last local model reported.
    supportsPreserveThinking: false,
    supportsTools: supportsStudioTools,
    supportsBuiltinWebSearch,
    supportsBuiltinCodeExecution,
    supportsBuiltinImageGeneration,
    supportsBuiltinWebFetch,
    toolsEnabled: nextToolsEnabled,
    codeToolsEnabled: canRunCode ? (storedCodeToolsEnabled ?? false) : false,
    imageToolsEnabled: supportsBuiltinImageGeneration
      ? (storedImageToolsEnabled ?? false)
      : false,
    // Default Fetch off (Anthropic bills per fetch); deliberate opt-in.
    webFetchToolsEnabled: supportsBuiltinWebFetch
      ? (storedWebFetchToolsEnabled ?? false)
      : false,
  };
}
