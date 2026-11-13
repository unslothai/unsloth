import assert from "node:assert/strict";
import test from "node:test";

import { atDefaultUiScale, readSrc } from "./helpers/kit.ts";

const PORTALLED_NATIVE_TITLEBAR_PATTERN =
  /"--studio-window-chrome-top",[\s\S]*?NATIVE_MAC_TITLEBAR_HEIGHT_VAR/;

test("mac titlebar navigation shifts buttons with centered glyphs", async () => {
  const [titlebar, provider] = await Promise.all([
    atDefaultUiScale(readSrc("components/tauri/window-titlebar.tsx")),
    atDefaultUiScale(readSrc("app/provider.tsx")),
  ]);
  const macStyle = provider.match(
    /const MAC_NATIVE_CHROME_STYLE = \{[\s\S]*?\} as CSSProperties;/,
  )?.[0];
  assert.ok(macStyle);
  assert.match(macStyle, /"--studio-titlebar-navigation-margin-top": "4px"/);
  assert.match(macStyle, /"--studio-titlebar-navigation-offset-y": "4px"/);
  assert.match(
    titlebar,
    /mt-\[var\(--studio-titlebar-navigation-margin-top,0px\)\]/,
  );

  assert.match(
    titlebar,
    /const iconClass = customTitlebar\s*\?\s*"size-\[18px\]"\s*:\s*"size-icon !size-\[calc\(var\(--icon-size\)\+1px\)\]";/,
  );
  assert.equal(titlebar.split("className={iconClass}").length - 1, 3);
});

test("mac chat and media headers share the lowered control row", async () => {
  const [provider, chat, images, video] = await Promise.all([
    atDefaultUiScale(readSrc("app/provider.tsx")),
    atDefaultUiScale(readSrc("features/chat/chat-page.tsx")),
    atDefaultUiScale(readSrc("features/images/images-page.tsx")),
    atDefaultUiScale(readSrc("features/video/video-page.tsx")),
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

// Assert the fallback is built from the same constant; retyped values diverge at other zooms.
test("mac native chrome clearance stays fixed across interface scales", async () => {
  const [provider, runtime] = await Promise.all([
    atDefaultUiScale(readSrc("app/provider.tsx")),
    atDefaultUiScale(readSrc("features/settings/lib/interface-scale-runtime.ts")),
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
  assert.doesNotMatch(provider, /--studio-native-titlebar-height, 34px/);
  assert.doesNotMatch(provider, /--studio-native-traffic-light-inset, 78px/);
  assert.match(provider, PORTALLED_NATIVE_TITLEBAR_PATTERN);
});
