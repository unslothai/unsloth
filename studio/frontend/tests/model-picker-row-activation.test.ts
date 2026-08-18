// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The On Device list renders every cached repo, and every row used to mount three Radix tooltips
// and a Radix dropdown whether or not anything had ever pointed at it. At 1000 downloaded models
// that was 3004 tooltip triggers and 1000 dropdown roots, and opening the picker took 5.9 s.
//
// The fix is invisible in the rendered output: the DOM node count is identical either way (5234
// panel nodes at 200 models on both sides, measured), only how much React and Radix machinery hangs
// off it differs. So, like thread-delete-render-budget.test.ts, the wiring is pinned at the source.
// Every seam below is silently load-bearing: undo any one of them and the picker still looks and
// behaves correctly in a hand test, and the list is slow again -- or, worse, it is fast and it has
// quietly stopped being keyboard reachable.
//
// The behaviour these seams produce is asserted for real, in a browser, by
// tests/studio/playwright_model_picker_deferred.py.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

const SELECTOR = "features/model-picker/components/model-selector";
const pickers = source(`${SELECTOR}/pickers.tsx`);
const activation = source(`${SELECTOR}/row-activation.tsx`);
const rowMenu = source(`${SELECTOR}/model-row-menu.tsx`);
const settingsAction = source(`${SELECTOR}/model-load-settings-action.tsx`);

// A declaration's body ends at the first `}` in column 0. Splitting on a bare "\n}" would stop at
// the closing brace of a destructured parameter list instead, which reads as "the seam is gone".
function block(text: string, start: string, end = "\n}\n"): string {
  const [, rest] = text.split(start, 2);
  assert.ok(rest !== undefined, `the source no longer contains ${start}`);
  const [body] = (rest ?? "").split(end, 1);
  return body ?? "";
}

test("option ids are looked up through a map, not scanned per row", () => {
  // getOptionProps runs once per ROW while the list renders, so an indexOf inside it is a linear
  // scan of the whole key list per row: O(N^2) string comparisons to draw N rows, half a million
  // of them at 1000 downloaded models.
  const roving = block(pickers, "function useRovingModelList(");
  assert.match(roving, /const optionIndexByKey = useMemo\(/);
  assert.match(roving, /const index = optionIndexByKey\.get\(optionKey\)/);
  const getOptionDomId = block(roving, "const getOptionDomId = useCallback(", "  );");
  assert.doesNotMatch(getOptionDomId, /optionKeys\.indexOf/);
});

test("the map keeps the first occurrence of a duplicated key, as indexOf did", () => {
  // Map.set would otherwise leave the LAST index behind and silently move a duplicated row's DOM
  // id, which is what the roving list focuses by.
  const roving = block(pickers, "function useRovingModelList(");
  assert.match(roving, /if \(key !== undefined && !byKey\.has\(key\)\) byKey\.set\(key, index\)/);
});

test("an On Device row stops before its tooltip when nothing has reached it", () => {
  // This early return is the win: the tooltip body, the Radix root, its portal and its Presence
  // state machine are all built after this point, and none of them is on screen for a row the
  // pointer has never been near.
  assert.match(pickers, /const rowActive = useRowActive\(\);/);
  assert.match(pickers, /if \(!rowActive\) \{\s*\n\s*return content;\s*\n\s*\}/);
});

test("every On Device row shell is a ModelRowShell", () => {
  // The shell is what turns activation on. A row rendered through a bare <div> keeps the merge
  // base's cost, so the count is pinned rather than the existence.
  const shells = pickers.match(/<ModelRowShell[\s>]/g) ?? [];
  assert.equal(shells.length, 5, `expected 5 On Device row shells, found ${shells.length}`);
  assert.doesNotMatch(pickers, /<div[^>]*className=\{downloadedRowShellClassName\(/);
});

test("a row that has not been reached still renders a real, labelled dots button", () => {
  // Same tag, same label, same aria a closed Radix trigger carries: the tab order and the
  // accessibility tree have to be identical, because they are the two things a screen reader and a
  // keyboard user can see that a screenshot cannot.
  const placeholder = block(rowMenu, "function ModelRowMenuPlaceholder(");
  assert.match(placeholder, /aria-haspopup="menu"/);
  assert.match(placeholder, /aria-expanded=\{false\}/);
  assert.match(placeholder, /data-state="closed"/);
  assert.match(placeholder, /data-slot="dropdown-menu-trigger"/);
  assert.match(placeholder, /aria-label=\{ariaLabel\}/);
});

test("pressing the placeholder opens the menu instead of swallowing the press", () => {
  // Radix's own trigger opens on pointerdown. A placeholder that only mounted the real menu would
  // eat the gesture that asked for it.
  const shell = block(rowMenu, "export function ModelRowMenu(props: ModelRowMenuProps) {");
  assert.match(shell, /setActivated\(true\);\s*\n\s*setOpen\(true\);/);
  assert.match(rowMenu, /<DropdownMenu open=\{open\} onOpenChange=\{onOpenChange\}>/);
});

test("the update-completion subscription outlives the menu", () => {
  // It is how a row learns that a managed update it started has finished. Moved into the lazily
  // mounted body it would be gone by the time the download it kicked off completes.
  const shell = block(rowMenu, "export function ModelRowMenu(props: ModelRowMenuProps) {");
  assert.match(shell, /subscribeJobListeners\("model", updateRepoId/);
  const live = block(rowMenu, "function ModelRowMenuLive({");
  assert.doesNotMatch(live, /subscribeJobListeners/);
});

test("the gear keeps its own click handler while its tooltip is deferred", () => {
  // The button is the same element either way; only the tooltip around it waits. A gear that
  // needed activating to be clickable would break the first click on a cold row.
  assert.match(settingsAction, /const rowActive = useRowActive\(\);/);
  assert.match(settingsAction, /if \(!rowActive\) return button;/);
  const button = block(settingsAction, "const button = (", "  );");
  assert.match(button, /onClick=\{\(e\) => \{/);
  assert.match(button, /onConfigure\(\);/);
});

test("activation replays the focus it moves", () => {
  // Activating replaces the element that has focus. Without this, a Tab into the list drops focus
  // to <body> and the list stops being keyboard reachable -- silently, and only for real users.
  assert.match(activation, /childIndexPath\(shell, focused\)/);
  assert.match(activation, /target\.focus\(\{ preventScroll: true \}\)/);
  assert.match(activation, /!shell\.contains\(document\.activeElement\)/);
});

test("activation replays the pointer move Radix needs, and only inside the row", () => {
  // Radix opens a tooltip on pointermove, not on pointerenter, so a pointer that enters a row and
  // holds still would never open anything. Replaying at a point that has since left the row would
  // announce a hover that is not happening, so the containment check is part of the seam.
  assert.match(activation, /new PointerEvent\("pointermove"/);
  assert.match(activation, /pointerType: "mouse"/);
  assert.match(activation, /if \(under && shell\.contains\(under\)\)/);
  assert.match(activation, /useLayoutEffect\(/);
});

test("coarse-pointer devices are opted out entirely", () => {
  // They show the row actions at all times and a tap is a pointerdown and a click on the same
  // node, so there is no hover to trade on and nothing to gain by swapping a subtree under a
  // finger. Starting active means those devices render exactly what the merge base rendered.
  assert.match(activation, /useState\(pointerIsCoarse\)/);
  assert.match(activation, /matchMedia\("\(hover: none\)"\)/);
  assert.match(activation, /matchMedia\("\(pointer: coarse\)"\)/);
  assert.match(activation, /if \(event\.pointerType === "touch"\) return;/);
});

test("the context default mounts everything, so no other list changes", () => {
  // Every row outside a ModelRowShell -- the Hub catalog, Recommended, the variant expander --
  // reads the default and behaves exactly as it did.
  assert.match(activation, /createContext\(true\)/);
});
