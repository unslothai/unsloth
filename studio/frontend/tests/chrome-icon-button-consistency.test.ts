import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path: string) =>
  readFile(new URL(`../src/${path}`, import.meta.url), "utf8");

const BUTTON_CLASS = /className="([^"]+)"/;
const WHITESPACE = /\s+/;
const RUN_SETTINGS_HEADER =
  /<div className="([^"]*h-\[var\(--studio-chat-header-height,48px\)\][^"]*)"/;

function buttonAtMarker(
  contents: string,
  marker: string,
  occurrence = 0,
): string {
  let markerIndex = -1;
  for (let index = 0; index <= occurrence; index += 1) {
    markerIndex = contents.indexOf(marker, markerIndex + 1);
  }
  if (markerIndex === -1) {
    throw new Error(`missing button marker: ${marker}`);
  }
  const start = contents.lastIndexOf("<button", markerIndex);
  const end = contents.indexOf("</button>", markerIndex);
  if (start === -1) {
    throw new Error(`missing button start: ${marker}`);
  }
  if (end === -1) {
    throw new Error(`missing button end: ${marker}`);
  }
  return contents.slice(start, end);
}

function buttonClasses(button: string): Set<string> {
  const classes = BUTTON_CLASS.exec(button)?.[1];
  if (!classes) {
    throw new Error("button has no static className");
  }
  return new Set(classes.split(WHITESPACE));
}

test("run settings uses one aligned toggle in both states", async () => {
  const [page, panel] = await Promise.all([
    source("features/chat/chat-page.tsx"),
    source("features/chat/chat-settings-sheet.tsx"),
  ]);

  const toggles = [
    buttonAtMarker(page, 'aria-label="Open run settings"'),
    buttonAtMarker(panel, 'aria-label="Close run settings"'),
  ];
  for (const toggle of toggles) {
    const classes = buttonClasses(toggle);
    assert.ok(classes.has("size-[30px]"));
    assert.ok(classes.has("rounded-[10px]"));
    assert.ok(!classes.has("rounded-full"));
  }
  const header = RUN_SETTINGS_HEADER.exec(panel);
  assert.ok(header);
  const headerClasses = new Set(header[1].split(WHITESPACE));
  assert.ok(
    headerClasses.has("pt-[var(--studio-chat-header-padding-top,11px)]"),
  );
  assert.ok(headerClasses.has("pr-[18px]"));
});

test("settings chrome uses the titlebar rounded-square hover shape", async () => {
  const [dialog, sidebar] = await Promise.all([
    source("features/settings/settings-dialog.tsx"),
    source("components/app-sidebar.tsx"),
  ]);

  const dialogMain = dialog.slice(dialog.indexOf("<main"));
  const dialogClose = buttonClasses(
    buttonAtMarker(
      dialogMain,
      'aria-label={t("settings.dialog.closeAriaLabel")}',
    ),
  );
  assert.ok(dialogClose.has("size-[30px]"));
  assert.ok(dialogClose.has("rounded-[10px]"));
  assert.ok(!dialogClose.has("rounded-full"));

  const settingsCog = sidebar.slice(sidebar.indexOf("settings cog; sibling"));
  const settingsCogClasses = buttonClasses(
    buttonAtMarker(settingsCog, 'aria-label={t("shell.navigation.settings")}'),
  );
  assert.ok(settingsCogClasses.has("size-[32px]"));
  assert.ok(settingsCogClasses.has("rounded-[10px]"));
  assert.ok(!settingsCogClasses.has("rounded-full"));
});
