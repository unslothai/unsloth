// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thread feature flags for staged rollout. The wiring stays in place so flipping a flag here is
// the only edit needed to re-enable.

// Grid-based reasoning collapse: the reasoning pane opens and closes by transitioning
// `grid-template-rows` from `0fr` to `1fr` instead of animating `height` from `0` to
// `--radix-collapsible-content-height`, and it uses a local collapsible primitive that never calls
// `getBoundingClientRect`.
//
// Both halves are required. The CSS variable is supplied by Radix's `CollapsibleContentImpl`,
// whose `useLayoutEffect` writes `transitionDuration: 0s` / `animationName: none`, reads
// `getBoundingClientRect()`, then writes the styles back -- on every `open` change,
// UNCONDITIONALLY, whether or not any stylesheet consumes the variable it produces. That read is a
// synchronous full-document layout, and Blink's layout is O(total layout objects) rather than
// O(dirty objects), so it is charged the whole thread. Swapping only the keyframes leaves the
// measurement exactly where it was; see `unmeasured-collapsible.tsx`.
//
// Scoped to the reasoning pane. `tool-group.tsx`, `tool-fallback.tsx` and the shared
// `components/ui/collapsible.tsx` default still animate height: `app-sidebar.tsx` keys its scroll
// fade off `onAnimationEnd` with `animationName === "collapsible-down" | "collapsible-up"`, which a
// transition does not fire, and changing three collapsibles at once would make an A/B on the
// reasoning toggle unattributable.
//
// OFF until the A/B has run. CONTRIBUTING-perf.md is explicit that a direction is not a result, and
// this one also changes the rendered DOM (one wrapper element inside the content), so it needs
// screenshots as well as the parity digest.
export const GRID_COLLAPSE_REASONING_ENABLED = false;
