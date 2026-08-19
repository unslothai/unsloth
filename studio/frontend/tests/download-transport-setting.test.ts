// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which transport downloads run on was a toggle on the Hub page and nothing else, so the choice
// lived in one browser and most people never found it. It is a row in Settings > General now,
// saved for the install.
//
// The rules pinned here: this browser's own choice wins, the install's setting is next, the
// default is unchanged at Auto, and the row offers both transports with the difference spelled
// out.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DEFAULT_TRANSPORT_MODE, TRANSPORT, pickTransportMode } = await import(
  "../src/features/hub/download-manager/constants.ts"
);

function read(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

const ROW = read("features/settings/components/download-transport-row.tsx");
const PREFERENCE = read(
  "features/hub/download-manager/transport-preference.ts",
);
const GENERAL_TAB = read("features/settings/tabs/general-tab.tsx");
const EN = read("i18n/locales/en.ts");

test("this browser's own choice beats the install setting", () => {
  assert.equal(pickTransportMode("xet", "http"), TRANSPORT.XET);
  assert.equal(pickTransportMode("http", "auto"), TRANSPORT.HTTP);
  assert.equal(pickTransportMode("auto", "http"), TRANSPORT.AUTO);
});

test("with no choice of its own the install setting decides", () => {
  // Someone who picked a transport in Settings gets it in a browser that has never chosen.
  assert.equal(pickTransportMode(null, "auto"), TRANSPORT.AUTO);
  assert.equal(pickTransportMode(null, "xet"), TRANSPORT.XET);
  assert.equal(pickTransportMode(undefined, "http"), TRANSPORT.HTTP);
});

test("junk on either side leaves the default alone", () => {
  // Auto, as before this row existed: adding a setting must not move an install nobody has
  // opened it on.
  assert.equal(pickTransportMode("ftp", "torrent"), TRANSPORT.AUTO);
  assert.equal(pickTransportMode(null, null), DEFAULT_TRANSPORT_MODE);
  assert.equal(DEFAULT_TRANSPORT_MODE, TRANSPORT.AUTO);
});

test("a download waits for the install setting before picking a transport", () => {
  // getTransportMode() answers from what is already known, so a download that read it before the
  // settings fetch landed would miss a transport picked in another browser.
  for (const source of [
    read("features/hub/download-manager/poll-loop.ts"),
    read("features/hub/download-manager/transport-conflict.ts"),
  ]) {
    assert.match(source, /await resolveTransportMode\(\)/);
    assert.ok(
      !/[^a-zA-Z]getTransportMode\(\)/.test(source),
      "a download start reads the preference without waiting for it",
    );
  }
});

test("choosing a transport saves it for the install too", () => {
  // Otherwise the same install in another browser keeps the old transport.
  assert.match(PREFERENCE, /updateDownloadTransportSettings\(next\)/);
  // The local write still comes first: it is what this browser's downloads read.
  assert.ok(
    PREFERENCE.indexOf("localStorage.setItem") <
      PREFERENCE.indexOf("updateDownloadTransportSettings(next)"),
  );
});

test("the General tab carries the transport row", () => {
  assert.match(GENERAL_TAB, /<DownloadTransportRow \/>/);
  assert.match(GENERAL_TAB, /settings\.general\.downloads\.sectionTitle/);
});

test("the row offers HTTPS and Xet, and says which one is in force", () => {
  for (const key of ["downloads.https", "downloads.xet", "downloads.auto"]) {
    assert.ok(ROW.includes(key), `${key} is missing from the row`);
  }
  // Xet with no hf_xet is shown as unavailable rather than silently downloading over HTTPS.
  assert.match(ROW, /xetAvailable === false/);
  assert.match(ROW, /autoResolvesTo/);
});

test("the copy explains the difference, not just the names", () => {
  const downloads = EN.slice(
    EN.indexOf("      downloads: {"),
    EN.indexOf("      uploads: {"),
  );
  assert.ok(downloads.length > 0, "the downloads copy moved");
  // Resuming and the cancel behaviour are the practical difference between the two.
  assert.match(downloads, /resumes/i);
  assert.match(downloads, /cancel/i);
  assert.match(downloads, /hf_xet/);
});
