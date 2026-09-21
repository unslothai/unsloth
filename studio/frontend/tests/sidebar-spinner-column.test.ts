// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const APP_SIDEBAR = readSrc("components/app-sidebar.tsx");

// The nav spinner is ml-auto, so it sits at its row's padding-right plus its margin-right.
// The chat spinner is anchored at right-N off the row's edge instead: in the flow a working
// row swapped pr-4 for pr-16 and shoved it 64px in, which reading pr-4 off the base class
// never saw. So measure the chat side from the edge it is anchored to.
const TAILWIND_UNIT = 4;

function inset(classes: string, prefix: string): number {
  const m = new RegExp(`(?:^| )${prefix}-([0-9.]+)(?: |$)`).exec(classes);
  return m ? Number(m[1]) * TAILWIND_UNIT : 0;
}

function grab(source: string, pattern: RegExp, what: string): string {
  const m = pattern.exec(source);
  assert.ok(m, `could not find ${what} in app-sidebar.tsx`);
  return m[1];
}

test("nav and Recents spinners land on one trailing column", async () => {

  const navRow = grab(
    APP_SIDEBAR,
    /className="(sidebar-nav-btn h-\[33px\] rounded-full[^"]*)"/,
    "NavItem row",
  );
  const navSpinner = grab(
    APP_SIDEBAR,
    /<Spinner className="(ml-auto[^"]*group-data-\[collapsible=icon\]:hidden)"/,
    "NavItem spinner",
  );
  // The wrapper the chat spinner hangs off, anchored to the row rather than its text box.
  const chatSpinnerAnchor = grab(
    APP_SIDEBAR,
    /className=\{cn\(\s*"(pointer-events-none absolute right-[0-9.]+[^"]*)"[\s\S]{0,1200}?data-testid="chat-row-spinner"/,
    "Recents chat spinner anchor",
  );

  const nav = inset(navRow, "pr") + inset(navSpinner, "mr");
  const chat = inset(chatSpinnerAnchor, "right");

  assert.equal(
    nav,
    chat,
    `nav spinner sits ${nav}px in, chat spinner ${chat}px`,
  );
  assert.equal(nav, 16);

  // The row's padding must not hold the chat spinner out: that coupling is the bug.
  assert.ok(
    !/data-testid="chat-row-spinner"[\s\S]{0,400}?className="ml-auto/.test(APP_SIDEBAR),
    "the chat spinner is back in the flow, where the row's padding-right moves it",
  );
});

// The kebab overlays the row's right edge, so a spinner row must pad past it.
test("a working Recents row clears the kebab on hover", async () => {
  const [source, css] = await Promise.all([
    readSrc("components/app-sidebar.tsx"),
    readSrc("index.css"),
  ]);

  const kebabInset =
    inset(grab(css, /\.sidebar-row-action \{\s*@apply ([^;]*);/, "row action"), "pr") +
    inset(grab(css, /\.sidebar-row-action-glyph \{\s*@apply ([^;]*);/, "action glyph"), "size");
  assert.equal(kebabInset, 30);

  // The row holds that room open at rest rather than on hover: a spinner sits against the same
  // edge the actions reveal over, so there is nothing to reclaim by waiting for the pointer.
  const working = grab(
    source,
    /A spinner glyph cannot truncate[\s\S]{0,240}?showWorkSpinner \? "pr-([0-9.]+)"/,
    "the showWorkSpinner padding",
  );
  assert.ok(
    Number(working) * TAILWIND_UNIT >= kebabInset,
    `${Number(working) * TAILWIND_UNIT}px padding, needs ${kebabInset}px to clear the kebab`,
  );

  // focus-visible reveals the actions without hover, so every row reserves room there too: one
  // padding for the project rows and one for every row of Pinned and Recents.
  const focusPads = [...source.matchAll(/:focus-visible\]\/[a-z-]+:pr-([0-9.]+)/g)].map(
    (m) => Number(m[1]) * TAILWIND_UNIT,
  );
  assert.equal(focusPads.length, 2, `expected 2 focus paddings, got ${focusPads.length}`);
  for (const pad of focusPads) {
    assert.ok(pad >= kebabInset, `${pad}px focus padding, needs ${kebabInset}px`);
  }
});

// That same column now carries a second meaning: a row whose capability has not been measured
// yet. On a Mac the platform store seeds chatOnly from the user agent, so Train and Video used
// to paint disabled (opacity-50, inert) from first load and only recover once /api/health
// answered -- indistinguishable from a measured "your machine cannot do this".
test("a pending row spins instead of blacking out", async () => {
  const { resolveNavRowState } = await import("../src/components/nav-row-state.ts");

  const pending = resolveNavRowState({
    disabled: true,
    tooltip: "Training needs an NVIDIA or AMD GPU.",
    pending: true,
  });
  assert.equal(pending.disabled, false, "the guessed gray-out still renders");
  assert.equal(pending.spinner, true, "nothing tells the user the check is still running");
  assert.equal(
    pending.tooltip,
    undefined,
    "a reason for a verdict nobody has reached yet is shown on hover",
  );
});

// Detection is a cold `import torch` and can run for minutes, long enough for a silent
// spinner to read as a hung row, so it says what it is waiting for. The disabled hint is
// still withheld: no verdict is in yet.
test("a pending row says what it is waiting for", async () => {
  const { resolveNavRowState } = await import("../src/components/nav-row-state.ts");

  const pending = resolveNavRowState({
    disabled: true,
    tooltip: "Training needs an NVIDIA or AMD GPU.",
    pending: true,
    pendingTooltip: "Checking this machine for training support...",
  });
  assert.equal(pending.spinner, true);
  assert.equal(pending.disabled, false);
  assert.equal(pending.tooltip, "Checking this machine for training support...");
});

// Having the tooltip is not the same as showing it. Both renderers hid one by default for
// an enabled row: SidebarMenuButton only on the collapsed rail, MoreMenuItem only when
// disabled. A pending row is enabled, so the explanation reached neither.
test("a pending row is marked so both renderers can show its tooltip", async () => {
  const { resolveNavRowState } = await import("../src/components/nav-row-state.ts");

  assert.equal(
    resolveNavRowState({ pending: true, pendingTooltip: "Checking..." }).pending,
    true,
  );
  assert.equal(resolveNavRowState({ disabled: true, tooltip: "x" }).pending, false);
  assert.equal(resolveNavRowState({ spinner: true }).pending, false);
});

test("the expanded row and the flyout both show a pending tooltip", async () => {
  const [sidebar, appSidebar] = await Promise.all([
    readSrc("components/ui/sidebar.tsx"),
    readSrc("components/app-sidebar.tsx"),
  ]);

  // The rail-only rule has to make an exception, or an enabled row is silent while expanded.
  assert.match(
    sidebar,
    /hidden=\{isMobile \|\| \(!isDisabled && !alwaysTooltip && state !== "collapsed"\)\}/,
    "SidebarMenuButton still hides every enabled row's tooltip while expanded",
  );
  assert.match(
    appSidebar,
    /alwaysTooltip=\{rowState\.pending\}/,
    "the inline rows never ask for the exception",
  );
  // The flyout's title is not conditional on the grey-out any more.
  assert.match(appSidebar, /^\s*title=\{tooltip\}$/m, "MoreMenuItem drops a pending title");
  assert.ok(
    !appSidebar.includes("title={disabled ? tooltip : undefined}"),
    "MoreMenuItem still gates its title on disabled",
  );
});

test("both capability rows carry a pending tooltip", async () => {
  for (const row of ["train", "video"]) {
    const block = APP_SIDEBAR.slice(APP_SIDEBAR.indexOf(`    ${row}: {`));
    const body = block.slice(0, block.indexOf("\n    },"));
    assert.match(
      body,
      /pendingTooltip: t\("shell\.navigation\.\w+Checking"\)/,
      `the ${row} row spins without saying why`,
    );
  }
});

test("a measured row is left exactly as it was", async () => {
  const { resolveNavRowState } = await import("../src/components/nav-row-state.ts");

  const measured = resolveNavRowState({
    disabled: true,
    tooltip: "Training needs MLX. Run `unsloth studio update` to enable Train.",
    spinner: false,
  });
  assert.equal(measured.disabled, true, "a real chat-only host was let into Train");
  assert.equal(measured.tooltip, "Training needs MLX. Run `unsloth studio update` to enable Train.");
  assert.equal(measured.spinner, false);

  // The pre-existing use of the column (a run in progress) is untouched.
  const working = resolveNavRowState({ spinner: true });
  assert.equal(working.spinner, true);
  assert.equal(working.disabled, undefined);
});

// Two render sites take these props: the inline rows and the More flyout. A row moved into
// More by Settings -> Appearance must not go back to rendering the guess.
test("both nav render sites resolve pending the same way", async () => {
  const resolves = APP_SIDEBAR.match(/const rowState = resolveNavRowState\(row\);/g) ?? [];
  assert.equal(resolves.length, 2, `expected both render sites to resolve, got ${resolves.length}`);
  // And neither passes the raw fields past it.
  for (const raw of ["disabled={row.disabled}", "tooltip={row.tooltip}", "spinner={row.spinner}"]) {
    assert.ok(!APP_SIDEBAR.includes(raw), `a render site still passes ${raw} unresolved`);
  }
});
