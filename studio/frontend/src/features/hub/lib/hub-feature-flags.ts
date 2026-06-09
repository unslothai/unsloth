// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub feature flags for staged rollout. Each gates a CTA whose integration ships
// in a later PR; JSX/wiring stay in place so flipping here is the only re-enable edit.

// Post-download Run / chat / Train CTAs, hidden until the Hub-aware chat and train pickers ship.
export const HUB_POST_DOWNLOAD_ACTIONS_VISIBLE = false;
