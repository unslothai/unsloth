// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Staged-rollout flags; flipping a flag here is the only edit needed.

// Reasoning pane collapses via grid-template-rows 0fr/1fr with a local primitive that never calls
// getBoundingClientRect: Radix measures layout on every open change, charging the whole thread.
// Other collapsibles still animate height (app-sidebar keys off onAnimationEnd).
export const GRID_COLLAPSE_REASONING_ENABLED = true;
