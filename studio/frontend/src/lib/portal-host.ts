// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One shared container for every Radix portal, so that React never attaches its delegated
 * event listeners to `<body>`.
 *
 * WHY THIS EXISTS, AND WHAT IT IS NOT
 *
 * React attaches its full delegated event listener set to the container of every portal, the
 * first time a portal targets it, and never removes it. Radix's `Portal` defaults its container
 * to `document.body`. So the first time ANY menu, tooltip, popover, dialog or select opens,
 * `document.body` acquires a second React event root -- and `<body>` is an ancestor of the whole
 * application, so from then on every event anywhere in the app is dispatched twice, once at
 * `#root` and once at `<body>`. Measured on the chat thread, `document.body` goes from 1 event
 * listener type to 85 the first time a menu opens, and stays there for the life of the tab.
 *
 * THIS WAS BUILT AS A PERFORMANCE FIX AND IT IS NOT ONE. It was measured for exactly that and
 * found to buy nothing. On chromium at 300K characters of thread (43,422 DOM nodes, 110 assistant
 * messages), fresh page per arm, medians of 3, the same 20-step scroll gesture after a menu:
 *
 *     without this change   1543.6 ms, 11 frames over 33 ms, 11 long tasks
 *     with this change      1527.0 ms, 11 frames over 33 ms, 11 long tasks
 *
 * and the trace's caller attribution is the same on both sides, 36 React commits at
 * react-dom_client.js:9077. The census does move, 85 listener types back to 1, so the change does
 * what it says. The cost simply was not coming from there: it is driven by the element under a
 * stationary cursor changing as content scrolls past it, which is live work and has nothing to do
 * with which container a portal uses. See the correction section in
 * tests/studio/playwright_heavy_thread.py.
 *
 * So the case for this file is hygiene, not speed: one app-wide React event root instead of two,
 * and floating surfaces that stop double-dispatching every event in the application. Do not cite
 * it in a performance claim.
 *
 * WHY A `display: contents` DIV AND NOT `#root`
 *
 * Portaling into `#root` instead would also avoid the body root, but it puts every floating
 * surface inside the application's own layout and stacking context, where any ancestor carrying
 * `transform`, `filter`, `backdrop-filter`, `perspective` or `contain` becomes the containing
 * block for `position: fixed` popper wrappers and clips them. This tree has all of those.
 *
 * A dedicated div appended to `<body>` keeps the portal content in exactly the position in the
 * box tree it already had, and `display: contents` means the host generates no box at all, so
 * its children participate in the body stacking context precisely as they did when Radix
 * appended them to `<body>` directly. React attaches its listener set to this div rather than to
 * `<body>`, and a div that is a leaf of `<body>` only ever sees events from the floating surface
 * inside it, which is a menu rather than a 43,422-node thread.
 */

const HOST_ID = "unsloth-portal-host";

let cached: HTMLElement | null = null;

/**
 * The shared portal container, created on first use.
 *
 * Returns `undefined` rather than throwing when there is no document, so that a caller can spread
 * it into a Radix `Portal` unconditionally: `container={undefined}` is what Radix already
 * defaults to, so server rendering and tests keep the old behaviour instead of crashing.
 *
 * The `isConnected` check is load-bearing rather than defensive. A test that swaps the document
 * body, and the thread fixture that mounts a fresh tree per case, both leave the cached node
 * detached, and a detached container renders the menu into nothing at all -- which looks exactly
 * like a menu that failed to open.
 */
export const getPortalHost = (): HTMLElement | undefined => {
  if (typeof document === "undefined") return undefined;
  if (cached?.isConnected) return cached;
  const existing = document.getElementById(HOST_ID);
  if (existing instanceof HTMLElement) {
    cached = existing;
    return cached;
  }
  const host = document.createElement("div");
  host.id = HOST_ID;
  // No box of its own: see the note above on stacking fidelity.
  host.style.display = "contents";
  document.body.appendChild(host);
  cached = host;
  return cached;
};
