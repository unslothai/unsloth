// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Panel-resize style-recalc flag, for staged rollout. An unregistered custom property inherits,
// so writing --sidebar-width / --studio-sidebar-live-width on an ancestor of the thread restyled
// the whole document every drag frame (57x more elements on a real thread). The fix writes each
// var into a subtree holding only its consumers; `@property { inherits: false }` would break
// consumers that are descendants of the written element.
export const PANEL_RESIZE_SCOPED_VARS_ENABLED = true;
