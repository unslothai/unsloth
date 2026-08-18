// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// its own plain module so the node suite can drive it: refresh-context-usage.ts pulls in the adapter

export type CountablePromptInput = {
  // messages the count would send, system prompt and canvas instruction included
  outboundMessageCount: number;
  // whether the request asks the server to render tool schemas and the action nudge
  toolsRequested: boolean;
};

// with neither, the template renders bare scaffolding: Phi-4-mini emits "<|assistant|>" alone,
// one token, describing nothing anyone put in the chat
export function hasCountablePrompt({
  outboundMessageCount,
  toolsRequested,
}: CountablePromptInput): boolean {
  return outboundMessageCount > 0 || toolsRequested;
}
