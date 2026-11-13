// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// JSX-free so the node test runner can load it; edit-panels.tsx adds the Components.

import {
  DELIVERY_NEEDS_FIRERED,
  EDIT_ADAPTERS,
  type EditAdapter,
} from "../edit-adapters";
import type { AudioToolPanel } from "./types";

export type EditPanelLogic = Omit<AudioToolPanel<null>, "Component"> & {
  adapter: EditAdapter;
};

function editPanelLogic(adapter: EditAdapter): EditPanelLogic {
  const family = adapter.family;
  return {
    id: `edit-${family}`,
    families: [family],
    workflows: ["edit"],
    title: "How edits apply",
    claims: adapter.claims,
    adapter,
    // DotTTS-MF is dots_tts too but cannot edit.
    appliesTo: (ctx) =>
      ctx.audioFamily === family &&
      ctx.audioWorkflows?.includes("edit") === true,
    initial: () => null,
    toRequest: () => ({}),
    validate: (_value, core) => {
      const edit = core.edit;
      if (!edit) return null;
      if (edit.mode === "delivery")
        return adapter.delivery ? null : DELIVERY_NEEDS_FIRERED;
      return adapter.validateWords(edit.transcript, edit.edited);
    },
  };
}

export const EDIT_PANEL_LOGIC: readonly EditPanelLogic[] =
  Object.values(EDIT_ADAPTERS).map(editPanelLogic);
