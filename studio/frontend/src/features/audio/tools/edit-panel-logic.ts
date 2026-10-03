// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Edit's model tools without their controls: one "How edits apply" panel per edit family, which
// claims the options the adapter sets itself and holds Generate back with the adapter's reason.
// Free of JSX so the node test runner can load it; edit-panels.tsx adds the Components.

import {
  DELIVERY_NEEDS_FIRERED,
  EDIT_ADAPTERS,
  type EditAdapter,
} from "../edit-adapters";
import type { AudioToolPanelLogic } from "./panel-logic";

export type EditPanelLogic = AudioToolPanelLogic<null> & {
  adapter: EditAdapter;
};

export function editPanelLogic(adapter: EditAdapter): EditPanelLogic {
  const family = adapter.family;
  return {
    id: `edit-${family}`,
    families: [family],
    workflows: ["edit"],
    title: "How edits apply",
    claims: adapter.claims,
    adapter,
    // The family alone is not enough: DotTTS-MF is dots_tts too but cannot edit.
    appliesTo: (ctx) =>
      ctx.audioFamily === family &&
      ctx.audioWorkflows?.includes("edit") === true,
    initial: () => null,
    // The edit part is built from the page's inputs by buildEditRun, not from a panel value.
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

/** One panel per edit adapter, in the page's model order. */
export const EDIT_PANEL_LOGIC: readonly EditPanelLogic[] = [
  editPanelLogic(EDIT_ADAPTERS.dots_tts),
  editPanelLogic(EDIT_ADAPTERS.vevo2),
  editPanelLogic(EDIT_ADAPTERS.firered_audio),
];
