// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Both spinners are ml-auto, so each one sits at its row's padding-right plus
// its own margin-right. The two rows carry different padding, so the margins
// have to make up the difference or the column visibly steps.
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
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );

  const navRow = grab(
    source,
    /className="(sidebar-nav-btn h-\[33px\] rounded-full[^"]*)"/,
    "NavItem row",
  );
  const navSpinner = grab(
    source,
    /<Spinner className="(ml-auto[^"]*group-data-\[collapsible=icon\]:hidden)"/,
    "NavItem spinner",
  );
  const chatRow = grab(
    source,
    // Row height is a density choice and moves independently of the trailing
    // column, so match any h-[Npx]; cursor-pointer is what makes this the chat row.
    /"(sidebar-nav-btn h-\[\d+px\] cursor-pointer rounded-full[^"]*)"/,
    "Recents chat row",
  );
  const chatSpinner = grab(
    source,
    /data-testid="chat-row-spinner"[\s\S]{0,400}?className="(ml-auto[^"]*)"/,
    "Recents chat spinner",
  );

  const nav = inset(navRow, "pr") + inset(navSpinner, "mr");
  const chat = inset(chatRow, "pr") + inset(chatSpinner, "mr");

  assert.equal(
    nav,
    chat,
    `nav spinner sits ${nav}px in, chat spinner ${chat}px`,
  );
  assert.equal(nav, 16);
});

// The kebab overlays the row's right edge, so a spinner row must pad past it.
test("a working Recents row clears the kebab on hover", async () => {
  const [source, css] = await Promise.all([
    readFile(new URL("../src/components/app-sidebar.tsx", import.meta.url), "utf8"),
    readFile(new URL("../src/index.css", import.meta.url), "utf8"),
  ]);

  const kebabInset =
    inset(grab(css, /\.sidebar-row-action \{\s*@apply ([^;]*);/, "row action"), "pr") +
    inset(grab(css, /\.sidebar-row-action-glyph \{\s*@apply ([^;]*);/, "action glyph"), "size");
  assert.equal(kebabInset, 30);

  const working = grab(
    source,
    // Anchor on the branch comment so nested spinner ternaries cannot redirect
    // the match to a pinned or project row.
    /A spinner glyph cannot truncate[\s\S]{0,120}?"(group-hover\/recent-item:pr-[^"]*)"/,
    "the showWorkSpinner padding branch",
  );
  // hover, menu-open and coarse-pointer all reveal the kebab, so all must clear it
  const pads = [...working.matchAll(/:pr-([0-9.]+)(?: |$)/g)].map(
    (m) => Number(m[1]) * TAILWIND_UNIT,
  );
  assert.ok(pads.length >= 3, `expected hover, open and coarse paddings, got ${pads.length}`);
  for (const pad of pads) {
    assert.ok(pad >= kebabInset, `${pad}px padding, needs ${kebabInset}px to clear the kebab`);
  }

  // focus-visible reveals the kebab without hover, so every spinner row reserves room there too.
  const focusPads = [...source.matchAll(/:focus-visible\]\/[a-z-]+:pr-([0-9.]+)/g)].map(
    (m) => Number(m[1]) * TAILWIND_UNIT,
  );
  assert.equal(focusPads.length, 3, `expected 3 focus paddings, got ${focusPads.length}`);
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
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const resolves = source.match(/const rowState = resolveNavRowState\(row\);/g) ?? [];
  assert.equal(resolves.length, 2, `expected both render sites to resolve, got ${resolves.length}`);
  // And neither passes the raw fields past it.
  for (const raw of ["disabled={row.disabled}", "tooltip={row.tooltip}", "spinner={row.spinner}"]) {
    assert.ok(!source.includes(raw), `a render site still passes ${raw} unresolved`);
  }
});
