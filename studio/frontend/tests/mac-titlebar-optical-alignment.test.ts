import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path: string) =>
  readFile(new URL(`../src/${path}`, import.meta.url), "utf8");
const NATIVE_TITLEBAR_HEIGHT_PATTERN =
  /const NATIVE_MAC_TITLEBAR_HEIGHT =\s*"var\(--studio-native-titlebar-height, 34px\)"/;
const NATIVE_TRAFFIC_LIGHT_INSET_PATTERN =
  /const NATIVE_MAC_TRAFFIC_LIGHT_INSET =\s*"var\(--studio-native-traffic-light-inset, 78px\)"/;
const PORTALLED_NATIVE_TITLEBAR_PATTERN =
  /"--studio-window-chrome-top",[\s\S]*?NATIVE_MAC_TITLEBAR_HEIGHT/;

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

test("mac native chrome clearance stays fixed across interface scales", async () => {
  const provider = await source("app/provider.tsx");
  assert.match(provider, NATIVE_TITLEBAR_HEIGHT_PATTERN);
  assert.match(provider, NATIVE_TRAFFIC_LIGHT_INSET_PATTERN);
  assert.match(provider, PORTALLED_NATIVE_TITLEBAR_PATTERN);
});
