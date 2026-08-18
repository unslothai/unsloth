// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The three sidebar menus a user actually opens.
//
// A modal Radix layer parks `pointer-events: none` on <body> for as long as it is
// open. `pointer-events` is INHERITED, so that one write invalidates computed style
// for the whole mounted subtree, and on a long thread that subtree is the thread.
// Measured on one packaged production build, same 220-message thread, same run: the
// action-bar More menu (already non-modal) opens and closes in 38.6 ms and the
// sidebar's "Chat options" menu, identical interaction, 382.6 ms.
//
// Going non-modal also drops the shield that was absorbing the dismissing click, so
// each converted menu mounts <MenuDismissGuard /> in its content. A menu that is
// non-modal WITHOUT the guard is the dangerous half of this change, not a lesser
// version of it: the press that dismisses the menu also fires whatever control it
// landed on. Both halves are pinned here, per menu, so a later edit cannot drop one.
//
// The fourth test is the trap this change was filed for. The two menus that look
// converted in features/chat/thread-sidebar.tsx are converted in a file NOTHING
// imports, so vite tree-shakes it and those menus never mount. A modality pin on an
// unreachable file is a vacuous pin, so each file checked here is required to have a
// live importer.

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

const read = (relative: string): string =>
  readFileSync(path.join(SRC, relative), "utf8");

const LINE_COMMENT = /^[ \t]*\/\/.*$/gm;
const BLOCK_COMMENT = /\/\*[\s\S]*?\*\//g;
const JSX_COMMENT = /\{\s*\/\*[\s\S]*?\*\/\s*\}/g;

/** Strip comments so a code sample inside a rationale cannot satisfy a check. */
function code(source: string): string {
  return source
    .replace(JSX_COMMENT, "")
    .replace(BLOCK_COMMENT, "")
    .replace(LINE_COMMENT, "");
}

/**
 * The `<DropdownMenu ...> ... </DropdownMenu>` element whose body contains `marker`,
 * by tag counting. `<DropdownMenuTrigger` and friends share the prefix, so the open
 * tag is matched only when followed by whitespace or `>`.
 *
 * `marker` must identify exactly one menu, and that is asserted rather than assumed:
 * a marker that matches two menus would let a check pass by reading the wrong one.
 */
function menuContaining(source: string, marker: string): string {
  const OPEN_TAG = /<DropdownMenu(?=[\s>])/;
  const CLOSE_TAG = "</DropdownMenu>";
  /** Slice from an opening `<DropdownMenu` at `start` to its own closing tag. */
  const element = (start: number): string => {
    let depth = 0;
    let i = start;
    while (i < source.length) {
      const rest = source.slice(i);
      const relOpen = rest.search(OPEN_TAG);
      const openAt = relOpen === -1 ? -1 : i + relOpen;
      const closeAt = source.indexOf(CLOSE_TAG, i);
      assert.notEqual(closeAt, -1, "unbalanced <DropdownMenu> in source");
      if (openAt !== -1 && openAt < closeAt) {
        depth++;
        i = openAt + "<DropdownMenu".length;
        continue;
      }
      depth--;
      i = closeAt + CLOSE_TAG.length;
      if (depth === 0) return source.slice(start, i);
    }
    throw new Error("unbalanced <DropdownMenu> in source");
  };

  const found: string[] = [];
  for (const match of source.matchAll(new RegExp(OPEN_TAG.source, "g"))) {
    const body = element(match.index);
    if (body.includes(marker)) found.push(body);
  }
  // A root nested inside another would make the outer one match the inner marker as
  // well, so keep only matches that contain no other match, and require that the
  // marker picked out exactly one menu. A marker matching two would let a check pass
  // by reading the wrong menu.
  const innermost = found.filter(
    (body) => !found.some((other) => other !== body && body.includes(other)),
  );
  assert.equal(
    innermost.length,
    1,
    `expected exactly one <DropdownMenu> containing ${JSON.stringify(marker)}, found ${innermost.length}`,
  );
  return innermost[0] as string;
}

/**
 * Body of a `function NAME(` / `const NAME = (` declaration, by brace matching, so a
 * check reads only the function it names.
 */
function declaration(source: string, name: string): string {
  const start = source.search(
    new RegExp(`(?:export\\s+)?(?:function|const)\\s+${name}\\b`),
  );
  assert.notEqual(start, -1, `${name} not found`);
  const paren = source.indexOf("(", start);
  assert.notEqual(paren, -1, `${name} has no parameter list`);
  let parens = 0;
  let afterParams = -1;
  for (let i = paren; i < source.length; i++) {
    if (source[i] === "(") parens++;
    else if (source[i] === ")") {
      parens--;
      if (parens === 0) {
        afterParams = i;
        break;
      }
    }
  }
  assert.notEqual(afterParams, -1, `unbalanced parentheses in ${name}`);
  const open = source.indexOf("{", afterParams);
  assert.notEqual(open, -1, `${name} has no body`);
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    if (source[i] === "{") depth++;
    else if (source[i] === "}") {
      depth--;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces in ${name}`);
}

const NON_MODAL = /<DropdownMenu\s+modal=\{false\}/;
const GUARD = /<MenuDismissGuard\s*\/>/;

const SIDEBAR = code(read("components/app-sidebar.tsx"));
const CHAT_PAGE = code(read("features/chat/chat-page.tsx"));

/** Both halves of the conversion, named so a failure says which one went. */
function assertConverted(menu: string, what: string): void {
  assert.match(
    menu,
    NON_MODAL,
    `${what} is back on the body modal layer; opening it invalidates computed style ` +
      "for the whole thread subtree, which is the 382.6 ms against 38.6 ms this change removed",
  );
  assert.match(
    menu,
    GUARD,
    `${what} is non-modal with no <MenuDismissGuard /> in its content, so the click ` +
      "that dismisses it also fires whatever control it landed on",
  );
}

test("the sidebar list-header menu is off the modal layer and guarded", () => {
  assertConverted(
    declaration(SIDEBAR, "renderSidebarHeaderMenu"),
    'the sidebar list-header "..." menu (Organize chats / sort)',
  );
});

test("the sidebar per-thread Chat options menu is off the modal layer and guarded", () => {
  assertConverted(
    menuContaining(SIDEBAR, 'aria-label="Chat options"'),
    'the sidebar per-thread "Chat options" menu',
  );
});

test("the project page per-chat Chat options menu is off the modal layer and guarded", () => {
  assertConverted(
    menuContaining(CHAT_PAGE, 'aria-label="Chat options"'),
    'the project page per-chat "Chat options" menu',
  );
});

/** Every .ts/.tsx under src/, so "who imports this" is asked of the whole tree. */
function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) sourceFiles(full, out);
    else if (/\.tsx?$/.test(entry)) out.push(full);
  }
  return out;
}

test("the pinned files are reachable, unlike thread-sidebar.tsx", () => {
  const files = sourceFiles(SRC);
  // A sweep that walked nothing would report every file unreachable and pass the
  // negative half below while failing the positive half for the wrong reason.
  assert.ok(files.length > 200, `only walked ${files.length} source files`);

  const importersOf = (relative: string): string[] => {
    const abs = path.join(SRC, relative);
    const stem = abs.replace(/\.tsx?$/, "");
    const aliased = `@/${path.relative(SRC, stem)}`;
    return files.filter((file) => {
      if (file === abs) return false;
      const source = readFileSync(file, "utf8");
      if (source.includes(`"${aliased}"`)) return true;
      const rel = path.relative(path.dirname(file), stem);
      const spec = rel.startsWith(".") ? rel : `./${rel}`;
      return source.includes(`"${spec}"`);
    });
  };

  for (const pinned of [
    "components/app-sidebar.tsx",
    "features/chat/chat-page.tsx",
  ]) {
    assert.ok(
      importersOf(pinned).length > 0,
      `${pinned} has no importer under src/, so vite drops it and the modality ` +
        "pins above are checking a file that never mounts",
    );
  }

  // The control. thread-sidebar.tsx carries two converted menus and NOTHING imports
  // it, which is why the eleven-menu count in #9051 is eight live menus. If this ever
  // gains an importer, its menus join the matrix and this test should be the thing
  // that says so.
  assert.deepEqual(
    importersOf("features/chat/thread-sidebar.tsx"),
    [],
    "thread-sidebar.tsx has gained an importer; its two menus now mount and need " +
      "the same dismissal matrix as the three pinned above",
  );
});
