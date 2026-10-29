// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatArtifactsStore } from "@/features/chat";
import { browserPanelAvailable } from "./panel-availability";
import { useBrowserStore } from "./store";

/** Without a chat wired to the browser (an overlay), stage the prompt for the visible composer. */
export function stageEditsPrompt(prompt: string): void {
  useChatArtifactsStore.getState().stageFixPrompt(prompt);
  if (!browserPanelAvailable()) useBrowserStore.getState().closePanel();
}
