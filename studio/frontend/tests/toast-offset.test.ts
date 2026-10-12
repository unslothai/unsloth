// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  BROWSER_PAGE_INSET_VAR,
  CHAT_SETTINGS_INSET_VAR,
  cornerInsetScope,
  cornerInsets,
  getToastOffsets,
  insetPastChatSettings,
  watchChatSettingsInset,
} from "../src/lib/toast-offset.ts";
import { readSrc } from "./helpers/kit.ts";

test("web chat toasts clear the header and stay against the right edge", () => {
  assert.deepEqual(getToastOffsets("/chat", false, false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/chat/thread", false, false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
});

test("web media toasts clear their workspace headers", () => {
  for (const pathname of ["/images", "/video"]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 52, right: 12 },
      mobile: { top: 52, right: 16 },
    });
  }
});

test("other web routes keep the normal corner inset", () => {
  for (const pathname of ["/studio", "/settings"]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("desktop routes without page headers clear the titlebar", () => {
  assert.deepEqual(getToastOffsets("/settings", true, false), {
    default: { top: 46, right: 12 },
    mobile: { top: 50, right: 16 },
  });
});

test("custom-titlebar desktop headers clear both titlebar bands", () => {
  for (const pathname of ["/chat", "/images", "/video", "/audio"]) {
    assert.deepEqual(getToastOffsets(pathname, true, true), {
      default: { top: 86, right: 12 },
      mobile: { top: 86, right: 16 },
    });
  }
});

test("macOS desktop headers overlay the native titlebar", () => {
  for (const pathname of ["/chat", "/images", "/video", "/audio"]) {
    assert.deepEqual(getToastOffsets(pathname, true, false), {
      default: { top: 52, right: 12 },
      mobile: { top: 52, right: 16 },
    });
  }
});

test("a route that merely starts with a workspace name keeps the corner inset", () => {
  // The header routes are matched exactly, so a longer path that happens to share the
  // prefix must not inherit their clearance and drop 40px down a page with no header.
  for (const pathname of ["/chatty", "/images-old", "/videos", "/chatgpt"]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("an unrecognised pathname falls back to the corner inset", () => {
  // The 404 shell paints no page header. This also covers a trailing-slash URL: the
  // router does not normalise it, so "/images/" rests as its own pathname and misses
  // the route, which is why it wants the no-header placement rather than the media one.
  for (const pathname of ["/unknown", "/images/", "/video/", ""]) {
    assert.deepEqual(getToastOffsets(pathname, false, false), {
      default: { top: 12, right: 12 },
      mobile: { top: 16, right: 16 },
    });
  }
});

test("a custom titlebar is ignored off the desktop app", () => {
  // shouldUseCustomWindowTitlebar() cannot return true while isTauri is false, but the
  // signature allows the pair, and there is no titlebar to clear in a browser.
  for (const pathname of ["/chat", "/studio"]) {
    assert.deepEqual(
      getToastOffsets(pathname, false, true),
      getToastOffsets(pathname, false, false),
    );
  }
});

test("offsets are pure, so a caller cannot poison the next lookup", () => {
  const first = getToastOffsets("/chat", false, false);
  first.default.top = -999;
  first.mobile.right = -999;
  assert.deepEqual(getToastOffsets("/chat", false, false), {
    default: { top: 52, right: 12 },
    mobile: { top: 52, right: 16 },
  });
});

test("the header offset follows the UI font size, the titlebar does not", () => {
  // The page header is 48px * the scale, so a fixed 52px top lands inside it
  // at the 20px setting. The titlebar band is fixed and keeps its 34px.
  assert.deepEqual(getToastOffsets("/chat", false, false, 20 / 15), {
    default: { top: 69, right: 12 },
    mobile: { top: 69, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/chat", true, true, 20 / 15), {
    default: { top: 103, right: 12 },
    mobile: { top: 103, right: 16 },
  });
  assert.deepEqual(getToastOffsets("/settings", false, false, 20 / 15), {
    default: { top: 12, right: 12 },
    mobile: { top: 16, right: 16 },
  });
});

test("desktop toasts shift left by the open Run settings panel", () => {
  assert.deepEqual(insetPastChatSettings({ top: 52, right: 12 }), {
    top: 52,
    right: "calc(12px + max(var(--studio-chat-settings-inset, 0px), var(--studio-browser-page-inset, 0px)))",
  });
});

function fakeInsetDom(rowWidth: number, panelWidth: number) {
  const vars = new Map<string, string>();
  const root = {
    style: {
      setProperty: (name: string, value: string) => void vars.set(name, value),
      removeProperty: (name: string) => {
        vars.delete(name);
        return "";
      },
    },
  };
  const row = { clientWidth: rowWidth };
  const panel = { offsetWidth: panelWidth, parentElement: row };
  const observed = new Set<object>();
  let notify = () => {};
  class Observer {
    constructor(callback: () => void) {
      notify = callback;
    }
    observe(target: object) {
      observed.add(target);
    }
    disconnect() {
      observed.clear();
    }
  }
  const resize = (target: object) => {
    if (observed.has(target)) notify();
  };
  return { vars, root, row, panel, Observer, resize };
}

test("the inset follows the panel while it is dragged wider", () => {
  const dom = fakeInsetDom(1400, 320);
  watchChatSettingsInset(dom.root, dom.panel, 320, 1, dom.Observer);
  assert.equal(dom.vars.get("--studio-chat-settings-inset"), "320px");

  dom.panel.offsetWidth = 520;
  dom.resize(dom.panel);
  assert.equal(dom.vars.get("--studio-chat-settings-inset"), "520px");
});

test("the inset is dropped when the chat column cannot hold a corner card", () => {
  const dom = fakeInsetDom(1400, 320);
  const stop = watchChatSettingsInset(
    dom.root,
    dom.panel,
    320,
    1,
    dom.Observer,
  );
  dom.row.clientWidth = 700;
  dom.resize(dom.row);
  assert.equal(dom.vars.has("--studio-chat-settings-inset"), false);

  dom.row.clientWidth = 1400;
  dom.resize(dom.row);
  assert.equal(dom.vars.get("--studio-chat-settings-inset"), "320px");

  stop();
  assert.equal(dom.vars.has("--studio-chat-settings-inset"), false);
});

test("a larger UI scale needs a wider chat column before the inset applies", () => {
  // 1100 - 320 = 780 holds a 448px card at scale 1 but not a 1.8x one (850px).
  const plain = fakeInsetDom(1100, 320);
  watchChatSettingsInset(plain.root, plain.panel, 320, 1, plain.Observer);
  assert.equal(plain.vars.get("--studio-chat-settings-inset"), "320px");

  const scaled = fakeInsetDom(1100, 320);
  watchChatSettingsInset(scaled.root, scaled.panel, 320, 1.8, scaled.Observer);
  assert.equal(scaled.vars.has("--studio-chat-settings-inset"), false);
});

test("a smaller UI scale still leaves room for the fixed-width download panel", () => {
  // 1024 - 280 sidebar = 744 row; 744 - 340 = 404 < 400 + 44, though 448 * 0.8 + 44 = 402.4 fits.
  const dom = fakeInsetDom(744, 340);
  watchChatSettingsInset(dom.root, dom.panel, 340, 0.8, dom.Observer);
  assert.equal(dom.vars.has("--studio-chat-settings-inset"), false);
});

function fakeScope() {
  const vars = new Map<string, string>();
  const element = {
    style: {
      setProperty: (name: string, value: string) => void vars.set(name, value),
      removeProperty: (name: string) => {
        vars.delete(name);
        return "";
      },
    },
  } as unknown as HTMLElement;
  return { vars, element };
}

test("corner insets reach every scope, including one that mounts later", (t) => {
  const toaster = fakeScope();
  const rail = fakeScope();
  t.after(cornerInsetScope(toaster.element) ?? (() => {}));
  t.after(cornerInsetScope(rail.element) ?? (() => {}));
  t.after(() => {
    cornerInsets.style.removeProperty(CHAT_SETTINGS_INSET_VAR);
    cornerInsets.style.removeProperty(BROWSER_PAGE_INSET_VAR);
  });

  cornerInsets.style.setProperty(CHAT_SETTINGS_INSET_VAR, "320px");
  assert.equal(toaster.vars.get(CHAT_SETTINGS_INSET_VAR), "320px");
  assert.equal(rail.vars.get(CHAT_SETTINGS_INSET_VAR), "320px");
  assert.equal(cornerInsets.style.getPropertyValue(CHAT_SETTINGS_INSET_VAR), "320px");

  const late = fakeScope();
  const release = cornerInsetScope(late.element);
  assert.equal(late.vars.get(CHAT_SETTINGS_INSET_VAR), "320px");
  release?.();

  cornerInsets.style.setProperty(BROWSER_PAGE_INSET_VAR, "600px");
  assert.equal(late.vars.has(BROWSER_PAGE_INSET_VAR), false, "a released scope is no longer written");
  assert.equal(toaster.vars.get(BROWSER_PAGE_INSET_VAR), "600px");

  cornerInsets.style.removeProperty(CHAT_SETTINGS_INSET_VAR);
  assert.equal(toaster.vars.has(CHAT_SETTINGS_INSET_VAR), false);
  assert.equal(rail.vars.has(CHAT_SETTINGS_INSET_VAR), false);
  assert.equal(cornerInsets.style.getPropertyValue(CHAT_SETTINGS_INSET_VAR), "");
});

test("the inset follows a drag through the corner scopes", (t) => {
  const scope = fakeScope();
  t.after(cornerInsetScope(scope.element) ?? (() => {}));
  const dom = fakeInsetDom(1400, 320);
  const stop = watchChatSettingsInset(cornerInsets, dom.panel, 320, 1, dom.Observer);
  t.after(stop);
  assert.equal(scope.vars.get(CHAT_SETTINGS_INSET_VAR), "320px");
  dom.panel.offsetWidth = 480;
  dom.resize(dom.panel);
  assert.equal(scope.vars.get(CHAT_SETTINGS_INSET_VAR), "480px");
  stop();
  assert.equal(scope.vars.has(CHAT_SETTINGS_INSET_VAR), false);
});

test("panel insets are written to their consumers, never to <html>", () => {
  const sheet = readSrc("features/chat/chat-settings-sheet.tsx");
  assert.match(sheet, /return watchChatSettingsInset\(\s*cornerInsets,\s*asideRef\.current,/, "an inherited variable on <html> restyled the whole document on every drag frame");
  assert.match(readSrc("features/browser/native-view.ts"), /const style = cornerInsets\.style;/);
  assert.match(readSrc("components/ui/sonner.tsx"), /<div\s+ref=\{cornerInsetScope\}\s+style=\{\{ display: "contents" \}\}/);
  assert.equal(readSrc("app/provider.tsx").match(/ref=\{cornerInsetScope\}\s*\/\/ Scrolls at the cap/g)?.length, 2, "both corner rails read the Run settings inset");
  for (const file of ["features/chat/chat-settings-sheet.tsx", "features/browser/native-view.ts", "lib/toast-offset.ts"]) {
    assert.equal(/documentElement\.style\.setProperty\((?:CHAT_SETTINGS_INSET_VAR|BROWSER_PAGE_INSET_VAR)/.test(readSrc(file)), false, file);
  }
});

