// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No store or barrel imports, so the rules are testable off a browser.

export interface ChatTemplateSeedState {
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
}

export type ChatTemplateSeed = Partial<ChatTemplateSeedState>;

/** Blank and absent are the same template: /load normalises "" to null before launching. */
function sameTemplate(a: string | null, b: string | null): boolean {
  return (a?.trim() ? a : null) === (b?.trim() ? b : null);
}

export function resolveChatTemplateSeed(options: {
  incoming: string | null | undefined;
  previous: ChatTemplateSeedState;
  hydratingExistingModel: boolean;
  seedLoadParams: boolean;
}): ChatTemplateSeed {
  const { incoming, previous, hydratingExistingModel, seedLoadParams } =
    options;
  if (incoming === undefined || !seedLoadParams) {
    return {};
  }
  const unseeded =
    previous.loadedChatTemplateOverride === null &&
    previous.chatTemplateOverride === null;
  if (hydratingExistingModel || unseeded) {
    return {
      chatTemplateOverride: incoming,
      loadedChatTemplateOverride: incoming,
    };
  }
  // A steady poll must not overwrite a staged edit.
  if (sameTemplate(incoming, previous.loadedChatTemplateOverride)) {
    return {};
  }
  // Baseline always advances; the control follows only while it is not dirty.
  const controlIsDirty = !sameTemplate(
    previous.chatTemplateOverride,
    previous.loadedChatTemplateOverride,
  );
  return {
    loadedChatTemplateOverride: incoming,
    ...(controlIsDirty ? {} : { chatTemplateOverride: incoming }),
  };
}
