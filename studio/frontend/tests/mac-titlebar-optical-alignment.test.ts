import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("mac titlebar navigation shifts buttons with centered glyphs", async () => {
  const [titlebar, provider] = await Promise.all([
    readSrc("components/tauri/window-titlebar.tsx"),
    readSrc("app/provider.tsx"),
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

  const enlargedIconClass =
    'className="size-icon !size-[calc(var(--icon-size)+1px)]"';
  assert.equal(titlebar.split(enlargedIconClass).length - 1, 3);
});

test("mac chat and media headers share the lowered control row", async () => {
  const [provider, chat, images, video] = await Promise.all([
    readSrc("app/provider.tsx"),
    readSrc("features/chat/chat-page.tsx"),
    readSrc("features/images/images-page.tsx"),
    readSrc("features/video/video-page.tsx"),
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
