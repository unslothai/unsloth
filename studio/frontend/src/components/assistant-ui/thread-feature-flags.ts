// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thread feature flags for staged rollout. JSX/wiring stay in place so flipping
// a flag here is the only edit needed to re-enable.

// Windowed message list: mount only the messages near the viewport instead of
// every message the thread has ever rendered. OFF until the autoscroll hook
// follows the virtualizer's `isAtEnd()` rather than the viewport's
// `scrollHeight`, and until select-all-copy reads the message store rather than
// the DOM. Both are prerequisites, not polish: with this on and either missing,
// a streaming thread stops following and a copied conversation silently loses
// every message that happened to be unmounted.
//
// While this is false the thread renders the exact `<ThreadPrimitive.Messages>`
// element it always has, so the flag-off DOM is unchanged.
export const THREAD_MESSAGE_VIRTUALIZATION_ENABLED = false;
