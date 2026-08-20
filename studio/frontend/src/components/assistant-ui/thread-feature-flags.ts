// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thread feature flags for staged rollout. The wiring stays in place so flipping a flag here is
// the only edit needed to re-enable.

// Viewport-gated syntax highlighting: a code block far from the viewport renders as PLAIN text
// (one token per line) instead of carrying its full set of Shiki spans, and is re-highlighted when
// it comes back within `CODE_HIGHLIGHT_BUFFER_PX`. At the 100K rung, 42,134 of the thread's
// standing spans come from 57 code blocks; all-plain is 2,850, a 93.2% reduction in the nodes this
// mechanism can reach.
//
// OFF until the A/B has run: the mechanism is measured layout-neutral and the plugin's incremental
// streaming path is unchanged, but no floor-cleared interaction result has been taken yet, and
// CONTRIBUTING-perf.md is explicit that a direction is not a result.
//
// While this is false `createCodePlugin` is called with no gate at all, so the plugin takes the
// exact code path it always has and the flag-off DOM is unchanged -- there is not even an
// attribute stamped on a token.
export const VIEWPORT_GATED_CODE_HIGHLIGHTING_ENABLED = false;
