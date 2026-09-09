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
// ON. The A/B this comment was waiting for has run, on the standard tier at the 100K rung, four
// repetitions across two independent waves, each wave carrying its own null control measured in
// band under the same load. `reasoning_toggle.open_ms` cleared all three gates in BOTH waves:
// -33.6% against a 15.2% floor, and -20.6% against an 11.8% floor. It is the only metric that did,
// and the jank index cleared in the first wave and fell under its floor in the second, so the claim
// here is narrow: opening a reasoning pane on a long thread gets meaningfully cheaper, and nothing
// else was shown to move.
//
// The screenshots this comment asked for were taken, on two production builds of the same commit
// differing only in this boolean. Collapsed, the two panes are byte-identical PNGs. Expanded, they
// differ in 25 pixels along one text baseline at a maximum channel delta of 2/255, which is
// antialiasing. `textContent` is byte-identical in both states, 23,965 characters expanded.
// Geometry matches to the sub-pixel: root 744x5937.38 and content 718x5896.25 on both arms, the
// grid row resolving `1fr` to exactly the height Radix measures.
//
// The parity digest does report a difference, and it is real rather than noise: the per-message
// structural signature is 209 characters shorter, entirely within `[data-slot="reasoning-root"]`.
// It is accounted for to the byte: -242 from the class list (Radix's eight animation and duration
// tokens replaced by four grid tokens), -8 from the dropped `style` attribute presence token, and
// +41 for the wrapper element this comment already predicted. No `aria-*` semantics change; the
// only ARIA delta is Radix's `radix-` id prefix disappearing, and the control relationship resolves
// on both arms.
export const GRID_COLLAPSE_REASONING_ENABLED = true;
