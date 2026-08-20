// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Panel-resize style-recalc flags, for staged rollout. Wiring stays in place so
// flipping a flag here is the only edit needed.
//
// WHY THIS EXISTS
//
// An UNREGISTERED CSS custom property inherits. Writing one on an element marks
// inherited style dirty for every descendant of that element, so a write on
// `<html>` restyles the whole document and a write on the sidebar wrapper
// restyles everything the wrapper contains -- which, on the chat route, is the
// thread and all of its Shiki token spans.
//
// `PanelResizeHandle` writes TWO such properties on every animation frame of a
// sidebar drag: `--sidebar-width` on `[data-slot="sidebar-wrapper"]`, and
// `--studio-sidebar-live-width` on `document.documentElement`. Both are
// ancestors of the thread. Measured on the heavy-thread harness (Chromium,
// `UpdateLayoutTree.elementCount` from a devtools trace, median of 10 writes):
//
//   thread          elements   --sidebar-width on wrapper   scoped (this flag)
//   27,873 chars       4,000    4,286 restyled /  67.9 ms    1 restyled / 0.02 ms
//   84,407 chars      11,805   12,693 restyled / 193.2 ms    1 restyled / 0.02 ms
//  140,987 chars      19,576   21,065 restyled / 335.0 ms    1 restyled / 0.02 ms
//  253,791 chars      35,119   37,811 restyled / 576.6 ms    1 restyled / 0.02 ms
//
// The unscoped write is linear in thread size (8.54x over an 8.78x growth in
// elements); the scoped write is flat. Two of them per frame at 60fps is not a
// slow drag, it is a stalled one.
//
// Those numbers come off the vite DEV server. Repeating the decisive pair with
// the built production stylesheet swapped in over the same DOM costs 49.1 ms
// rather than 133.3 ms for the same 12,026 elements, so read the milliseconds
// above as roughly 2.7x high. The mechanism is not a dev artefact: the ratio
// between an unregistered write and an `inherits: false` one is 1333x on dev
// and 982x on prod, and `elementCount` is a property of the DOM and identical
// on both.
//
// End to end, driving the real handle through a real pointer drag over a
// 12,026-element document (8 drag frames):
//
//   flag off   12,022 elements restyled per frame, 1,088.25 ms total
//   flag on        12 elements restyled per frame,     3.76 ms total
//
// 12,022 of 12,026 is 99.97% of the document restyled to move one panel edge.
// Rendered geometry is identical at all 8 drag positions in both states, and
// --sidebar-width resolves to the same value at every consumer.
//
// WHY NOT `@property { inherits: false }`
//
// That is the right fix for a property whose only consumer is the element it is
// written to, which is why it works for `--aui-scroll-stabilizer`. It is the
// WRONG fix for both of these, and would silently break them:
//
//   * `--sidebar-width` is read by `[data-slot="sidebar"]`, `sidebar-gap` and
//     `sidebar-container` -- descendants. With `inherits: false` they would all
//     resolve it to the initial value and the sidebar would lose its width.
//   * `--studio-sidebar-live-width` is read by `window-titlebar.tsx`, also a
//     descendant of the `<html>` it is written on.
//
// Registration is not the mechanism. INHERITANCE is: a probe registered with
// `inherits: true` restyled all 37,817 elements in 576 ms, exactly like the
// unregistered one, while the same probe registered `inherits: false` restyled
// 1 element in 0.08 ms. So the fix here is to move each write DOWN to a subtree
// that actually consumes it, which is what this flag does:
//
//   * `--sidebar-width` -> `[data-slot="sidebar"]`, which contains every
//     consumer and does not contain the thread.
//   * `--studio-sidebar-live-width` -> the titlebar roots themselves. When no
//     custom titlebar is mounted (every non-Tauri build) there is no consumer
//     at all and the write is skipped entirely.
//
// NOT ADDRESSED BY THIS FLAG, and measured so the next person does not have to:
// `html[data-panel-resizing] *` (index.css:1604-1605) is a universal descendant selector
// on `<html>`, and toggling that attribute restyles the whole document too --
// 37,818 elements / 575.0 ms at 253,791 chars, indistinguishable from the
// custom-property write. It fires twice per drag rather than once per frame, so
// it is a smaller total, and it is load-bearing for cursor correctness. It is a
// separate change with its own visual risk, deliberately not bundled here.
export const PANEL_RESIZE_SCOPED_VARS_ENABLED = false;
