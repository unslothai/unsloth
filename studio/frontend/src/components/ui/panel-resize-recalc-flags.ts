// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Panel-resize style-recalc flags, for staged rollout. The wiring stays in
// place, so flipping a flag here is the only edit needed.
//
// WHY. An unregistered CSS custom property inherits, so writing one marks
// inherited style dirty for every descendant. `PanelResizeHandle` writes two of
// them on every animation frame of a sidebar drag -- `--sidebar-width` on
// `[data-slot="sidebar-wrapper"]` and `--studio-sidebar-live-width` on
// `document.documentElement` -- and both are ancestors of the chat thread, so
// each one restyles the whole document.
//
// Measured on the heavy-thread harness (Chromium, `UpdateLayoutTree.elementCount`
// from a devtools trace, median of 10 writes): the unscoped write is linear in
// thread size, 4,286 elements / 67.9 ms at a 4,000-element thread up to 37,811 /
// 576.6 ms at 35,119, while the scoped write is flat at 1 element / 0.02 ms.
// End to end, driving the real handle through a real pointer drag over a
// 12,026-element document (8 frames), 12,022 elements restyled per frame and
// 1,088.25 ms total become 12 per frame and 3.76 ms, with identical rendered
// geometry at all 8 positions and --sidebar-width resolving the same at every
// consumer.
//
// Read those milliseconds as roughly 2.7x high: they come off the vite DEV
// server, and the built production stylesheet costs 49.1 ms against 133.3 ms
// over the same DOM. The mechanism is not a dev artefact -- unregistered
// against `inherits: false` is 1333x on dev and 982x on prod, and elementCount
// is a DOM property, identical on both.
//
// WHY NOT `@property { inherits: false }`. That is right only for a property
// whose sole consumer is the element written to (`--aui-scroll-stabilizer`).
// Here it would silently break both: `--sidebar-width` is read by
// `[data-slot="sidebar"]`, `sidebar-gap` and `sidebar-container`, and
// `--studio-sidebar-live-width` by `window-titlebar.tsx`, all descendants of
// the element written to, so they would resolve the initial value instead.
// Registration is not the mechanism, INHERITANCE is: the same probe registered
// `inherits: true` restyled all 37,817 elements in 576 ms, `inherits: false`
// restyled 1 in 0.08 ms. So the fix moves each write DOWN into a subtree that
// consumes it -- `--sidebar-width` to `[data-slot="sidebar"]`, which holds
// every consumer and not the thread, and `--studio-sidebar-live-width` to the
// titlebar roots, which do not exist on a build without a custom titlebar,
// where the write is skipped entirely.
//
// NOT ADDRESSED BY THIS FLAG, measured so the next person need not:
// `html[data-panel-resizing] *` (index.css:1604-1605) is a universal descendant
// selector on `<html>` and restyles the whole document too, 37,818 elements /
// 575.0 ms. It fires twice per drag rather than once per frame, so the total is
// smaller, and it is load-bearing for cursor correctness: a separate change
// with its own visual risk, deliberately not bundled here.
export const PANEL_RESIZE_SCOPED_VARS_ENABLED = false;
