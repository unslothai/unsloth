// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Their own module, with no imports: the call sites that mark a subtree are ordinary chat and
// settings components, and a constant should not pull the whole index into their chunk.

/** Marks a subtree the bar must not read: the bar itself, first of all. */
export const FIND_SKIP_ATTRIBUTE = "data-find-skip";

/** Marks the scope the bar searches, set on the shell's content region in `__root.tsx`. */
export const FIND_SCOPE_ATTRIBUTE = "data-find-scope";
