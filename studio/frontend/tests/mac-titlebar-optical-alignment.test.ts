import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path: string) =>
  readFile(new URL(`../src/${path}`, import.meta.url), "utf8");
const PORTALLED_NATIVE_TITLEBAR_PATTERN =
  /"--studio-window-chrome-top",[\s\S]*?NATIVE_MAC_TITLEBAR_HEIGHT_VAR/;

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

// The runtime divides these by the zoom and provider.tsx uses them as CSS fallbacks. Two
// copies of 34 would agree at 100% and silently disagree everywhere else, so assert the
// fallback string is built from the same constant rather than retyped.
test("mac native chrome clearance stays fixed across interface scales", async () => {
  const [provider, runtime] = await Promise.all([
    source("app/provider.tsx"),
    source("features/settings/lib/interface-scale-runtime.ts"),
  ]);
  assert.match(
    runtime,
    /NATIVE_MAC_TITLEBAR_HEIGHT_VAR = `var\(--studio-native-titlebar-height, \$\{NATIVE_MAC_TITLEBAR_HEIGHT_PX\}px\)`/,
  );
  assert.match(
    runtime,
    /NATIVE_MAC_TRAFFIC_LIGHT_INSET_VAR = `var\(--studio-native-traffic-light-inset, \$\{NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX\}px\)`/,
  );
  assert.match(
    runtime,
    /NATIVE_MAC_TITLEBAR_HEIGHT_PX \/ zoom/,
  );
  assert.match(
    runtime,
    /NATIVE_MAC_TRAFFIC_LIGHT_INSET_PX \/ zoom/,
  );
  // No literal 34px or 78px left in provider.tsx to drift out from under the runtime.
  assert.doesNotMatch(provider, /--studio-native-titlebar-height, 34px/);
  assert.doesNotMatch(provider, /--studio-native-traffic-light-inset, 78px/);
  assert.match(provider, PORTALLED_NATIVE_TITLEBAR_PATTERN);
});
