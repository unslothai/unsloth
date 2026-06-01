// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub feature flags for staged rollout. Each flag gates a CTA that depends on
// integration code shipping in a later PR (Hub-aware chat picker, Hub-aware
// train picker, per-model config). The JSX and wiring stay in place so flips
// here are the only edit needed to re-enable.

// Post-download Run / "New Chat" / "Use in chat" / Train CTAs on inventory
// cards and the inspector panel. Hidden until the Hub-aware chat picker and
// Hub-aware train picker ship.
export const HUB_POST_DOWNLOAD_ACTIONS_VISIBLE = false;
