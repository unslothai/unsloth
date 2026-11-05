// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** On the floating preview's surface, as chat-page.tsx renders it. */
export const FLOATING_ATTRIBUTE = "data-browser-floating";
/** Fired on window per frame while the preview moves: a native page under it follows (native-view.ts). */
export const FLOATING_MOVE_EVENT = "unsloth-browser-floating-move";
/** On that surface: the page's scale, its side-pane layout width shown in the preview's width. */
export const FLOATING_SCALE_VAR = "--browser-floating-scale";
