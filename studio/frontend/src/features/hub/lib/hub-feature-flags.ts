// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub feature flags for staged rollout. JSX/wiring stay in place so flipping
// a flag here is the only edit needed to re-enable.

// Post-download Run / New Chat / Use in chat / Train CTAs; hidden until the
// Hub-aware chat and train pickers ship.
export const HUB_POST_DOWNLOAD_ACTIONS_VISIBLE = false;

export const HUB_GGUF_RUN_ACTIONS_VISIBLE = true;
