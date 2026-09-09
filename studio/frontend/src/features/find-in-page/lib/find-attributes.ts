// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import-free so ordinary call sites do not pull the index into their chunks.

/** Marks a subtree the bar must not read: the bar itself, first of all. */
export const FIND_SKIP_ATTRIBUTE = "data-find-skip";

/** Marks a visible panel portaled outside the shell scope as searchable. */
export const FIND_PORTAL_ATTRIBUTE = "data-find-portal";

/** Marks the scope the bar searches, set on the shell's content region in `__root.tsx`. */
export const FIND_SCOPE_ATTRIBUTE = "data-find-scope";
