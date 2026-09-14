// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a fresh /api/inference/status does to the chat-template override pair. Its own module,
// importing neither the store nor a barrel, so the rules are testable off a browser.

export interface ChatTemplateSeedState {
  /** The editable control: what the next load or Apply would send. */
  chatTemplateOverride: string | null;
  /** What the resident server was launched with, as this tab last saw it. */
  loadedChatTemplateOverride: string | null;
}

/** The subset of the pair to write back; empty means leave the store alone. */
export type ChatTemplateSeed = Partial<ChatTemplateSeedState>;

/** Blank and absent are the same template: /load normalises "" to null before launching. */
function sameTemplate(a: string | null, b: string | null): boolean {
  return (a?.trim() ? a : null) === (b?.trim() ? b : null);
}

export function resolveChatTemplateSeed(options: {
  /** ``status.chat_template_override``; undefined on a backend that omits the field. */
  incoming: string | null | undefined;
  previous: ChatTemplateSeedState;
  /** A different model or quant is being adopted, so both fields are stale. */
  hydratingExistingModel: boolean;
  /** No load of this tab's own is in flight (``!modelLoading``). */
  seedLoadParams: boolean;
}): ChatTemplateSeed {
  const { incoming, previous, hydratingExistingModel, seedLoadParams } =
    options;
  // While a load is in flight performLoad owns the load params, and an older backend omitting the
  // field says nothing about the template.
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
  // Same checkpoint and quant, steady template: an ordinary poll must not touch the control, or it
  // would overwrite a staged edit every refresh.
  if (sameTemplate(incoming, previous.loadedChatTemplateOverride)) {
    return {};
  }
  // The resident template moved under this tab: another tab or an API caller reloaded the same
  // checkpoint with a different override. The baseline is a fact about the running server, so it
  // always advances, since the rollback resends it. The control follows only while it still sits
  // on the old baseline; a genuinely dirty one keeps its intent, as the GPU group does.
  const controlIsDirty = !sameTemplate(
    previous.chatTemplateOverride,
    previous.loadedChatTemplateOverride,
  );
  return {
    loadedChatTemplateOverride: incoming,
    ...(controlIsDirty ? {} : { chatTemplateOverride: incoming }),
  };
}
