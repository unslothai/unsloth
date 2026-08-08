import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path: string) =>
  readFile(new URL(`../src/${path}`, import.meta.url), "utf8");

test("mac titlebar navigation shifts buttons with centered glyphs", async () => {
  const [titlebar, provider] = await Promise.all([
    source("components/tauri/window-titlebar.tsx"),
    source("app/provider.tsx"),
  ]);
  const macStyle = provider.match(
    /const MAC_NATIVE_CHROME_STYLE = \{[\s\S]*?\} as CSSProperties;/,
  )?.[0];
  assert.ok(macStyle);
  assert.match(macStyle, /"--studio-titlebar-navigation-offset-y": "4px"/);

  const enlargedIconClass =
    'className="size-icon !size-[calc(var(--icon-size)+1px)]"';
  assert.equal(titlebar.split(enlargedIconClass).length - 1, 3);
});

test("mac chat and media headers share the lowered control row", async () => {
  const [provider, chat, images, video] = await Promise.all([
    source("app/provider.tsx"),
    source("features/chat/chat-page.tsx"),
    source("features/images/images-page.tsx"),
    source("features/video/video-page.tsx"),
  ]);

  const macStyle = provider.match(
    /const MAC_NATIVE_CHROME_STYLE = \{[\s\S]*?\} as CSSProperties;/,
  )?.[0];
  assert.ok(macStyle);
  assert.match(macStyle, /"--studio-chat-header-padding-top": "9px"/);
  for (const page of [chat, images, video]) {
    assert.match(page, /var\(--studio-chat-header-padding-top,/);
  }
});
