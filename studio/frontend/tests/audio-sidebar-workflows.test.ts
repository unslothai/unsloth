// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SIDEBAR = readSrc("components/app-sidebar.tsx");

function block(source: string, start: string, end: string): string {
  const from = source.indexOf(start);
  assert.notEqual(from, -1, `missing ${start}`);
  const to = source.indexOf(end, from + start.length);
  assert.notEqual(to, -1, `missing ${end} after ${start}`);
  return source.slice(from, to);
}

test("Images keeps its list and chevron, now thin wrappers over the shared media pieces", () => {
  const images = block(
    SIDEBAR,
    "function ImagesNavDisclosure()",
    "function AudioNavDisclosure()",
  );
  assert.match(
    images,
    /<MediaNavDisclosure[\s\S]*revealClassName=\{IMAGES_DISCLOSURE_REVEAL\}/,
  );
  assert.match(
    images,
    /function ImagesWorkflowList\(\{\s*active,\s*collapsed,\s*onPick,\s*\}/,
  );
  assert.match(images, /tabs=\{WORKFLOW_TABS\}/);
  assert.match(
    images,
    /enabled=\{\(id\) => isWorkflowEnabled\(id, supported\)\}/,
  );
  assert.match(
    images,
    /listed=\{!collapsed && \(active \? pageMode !== "train" : expanded\)\}/,
  );
  assert.match(
    images,
    /const current = active && pageMode === "create" \? workflow : null;/,
  );
  // The reveal classes keep their literal spelling so Tailwind still generates them.
  assert.match(
    SIDEBAR,
    /const IMAGES_DISCLOSURE_REVEAL =\s*"group-hover\/images-item:opacity-100 group-hover\/images-item:pointer-events-auto";/,
  );
  assert.match(
    SIDEBAR,
    /const AUDIO_DISCLOSURE_REVEAL =\s*"group-hover\/audio-item:opacity-100 group-hover\/audio-item:pointer-events-auto";/,
  );
});

test("the Audio list highlights the requested or committed workflow, and never disables a row", () => {
  const audio = block(
    SIDEBAR,
    "function AudioWorkflowList(",
    "// Hugeicons' three dots",
  );
  assert.match(
    audio,
    /useAudioWorkspaceStore\(\(s\) => s\.requestedWorkflow\)/,
  );
  assert.match(
    audio,
    /const current = active \? \(requested \?\? workflow\) : null;/,
  );
  assert.match(audio, /tabs=\{AUDIO_WORKFLOWS\}/);
  assert.match(audio, /enabled=\{audioWorkflowAlwaysEnabled\}/);
  assert.match(SIDEBAR, /const audioWorkflowAlwaysEnabled = \(\) => true;/);
  assert.match(audio, /listed=\{!collapsed && \(active \|\| expanded\)\}/);
});

test("a pinned Audio row lists its workflows and hands picks to the page as requests", () => {
  assert.match(SIDEBAR, /id === "audio" && "group\/audio-item"/);
  assert.match(SIDEBAR, /const audioWorkflowsListed = sidebarRowsLabelled;/);
  assert.match(
    SIDEBAR,
    /\(id === "images" && imagesWorkflowsListed\) \|\|\s*\(id === "audio" && audioWorkflowsListed\)\s*\? false\s*: row\.active/,
  );
  assert.match(
    SIDEBAR,
    /id === "audio" &&\s*!row\.active &&\s*sidebarRowsLabelled \? \(\s*<AudioNavDisclosure \/>/,
  );
  assert.match(
    SIDEBAR,
    /<AudioWorkflowList\s+active=\{row\.active\}\s+collapsed=\{!sidebarRowsLabelled\}\s+onPick=\{pickAudioWorkflow\}/,
  );
  const pick = block(SIDEBAR, "const pickAudioWorkflow =", "};");
  assert.match(
    pick,
    /useAudioWorkspaceStore\.getState\(\)\.requestWorkflow\(workflowId\);/,
  );
  assert.match(pick, /navigate\(\{ to: "\/audio" \}\);/);
  assert.match(pick, /closeMobileIfOpen\(\);/);
  assert.doesNotMatch(SIDEBAR, /commitWorkflow/);
});

test("More opens Audio's workflows in a submenu that keeps the flyout open on the way in", () => {
  const more = block(SIDEBAR, "{overflowNavIds.map((id) => {", "<MoreMenuItem");
  assert.match(
    more,
    /if \(id === "audio"\) \{\s*return \(\s*<AudioMoreSubmenu/,
  );
  assert.match(more, /disabled=\{rowState\.disabled\}/);
  assert.match(more, /\.\.\.sidebarSubmenuOffsets,/);
  assert.match(more, /\.\.\.moreHover\.content,/);
  assert.match(more, /onPick=\{pickAudioWorkflow\}/);
  const resolves =
    SIDEBAR.match(/const rowState = resolveNavRowState\(row\);/g) ?? [];
  assert.equal(resolves.length, 2);

  const submenu = block(
    SIDEBAR,
    "function AudioMoreSubmenu(",
    "export function AppSidebar()",
  );
  assert.match(submenu, /<DropdownMenuSubTrigger/);
  assert.match(
    submenu,
    /className=\{cn\("gap-2\.5", active && "bg-accent\/60"\)\}/,
  );
  assert.match(submenu, /<DropdownMenuSubContent \{\.\.\.contentProps\}/);
  assert.match(submenu, /AUDIO_WORKFLOWS\.map/);
  assert.match(submenu, /onSelect=\{\(\) => onPick\(tab\.id\)\}/);
  assert.match(submenu, /current === tab\.id && "bg-accent\/60"/);
  assert.doesNotMatch(submenu, /max-h/);
});

test("clicking Audio in More opens the page, and the open page keeps the row out of More", () => {
  const submenu = block(
    SIDEBAR,
    "function AudioMoreSubmenu(",
    "export function AppSidebar()",
  );
  // preventing the click stops Radix from opening the submenu; keys keep the parent behavior.
  assert.match(
    submenu,
    /onClick=\{\(event\) => \{\s*if \(disabled\) return;\s*event\.preventDefault\(\);\s*onOpen\(\);\s*\}\}/,
  );
  assert.doesNotMatch(submenu, /onKeyDown/);

  const more = block(SIDEBAR, "{overflowNavIds.map((id) => {", "<MoreMenuItem");
  assert.match(
    more,
    /onOpen=\{\(\) => \{\s*setMoreOpen\(false\);\s*row\.onClick\(\);\s*\}\}/,
  );
  // omitting a workflow request preserves the page's current workflow.
  assert.match(
    SIDEBAR,
    /audio: \{[\s\S]*?onClick: \(\) => \{\s*navigateFromRow\(\{ to: "\/audio" \}\);\s*closeMobileIfOpen\(\);\s*\}/,
  );

  assert.match(
    SIDEBAR,
    /const \{ inline: inlineNavIds, overflow: overflowNavIds \} = placeNavRows\(\s*sidebarNav\.map\(\(item\) => \(\{ id: item\.id, pinned: navRowPinned\(item\) \}\)\),\s*navRows\.audio\.active \? "audio" : null,\s*\);/,
  );
});

test("the Audio row stays lit while hovered, like Images", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /\n\t\.group\\\/audio-item:hover > \.relative > \.sidebar-nav-btn,/,
  );
  assert.match(
    css,
    /\.dark \.group\\\/audio-item:hover > \.relative > \.sidebar-nav-btn,/,
  );
});

test("a nav row's New pill sits beside its label, clear of the trailing disclosure", () => {
  const item = block(
    SIDEBAR,
    "function NavItem(",
    "const WORKFLOW_UNAVAILABLE",
  );
  // no ml-auto: a trailing pill would sit under the overlay's chevron.
  assert.match(
    item,
    /<Badge\s+variant="secondary"\s+className="group-data-\[collapsible=icon\]:hidden"\s*>\s*\{badge\}/,
  );
  assert.match(item, /\{overlay\}/);
  const more = block(
    SIDEBAR,
    "{overflowNavIds.map((id) => {",
    "<DropdownMenuSeparator",
  );
  assert.equal(more.match(/badge=\{row\.badge\}/g)?.length, 2);
});

test("Audio is still unpinned by default", () => {
  const store = readSrc("features/settings/stores/appearance-custom-store.ts");
  assert.match(
    store,
    /SIDEBAR_NAV_DEFAULT_PINNED[^=]*= \{[^}]*\n\s*audio: false,/,
  );
});
