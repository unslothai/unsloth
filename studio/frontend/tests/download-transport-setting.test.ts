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
const TOGGLE = read("features/hub/catalog/transport-toggle.tsx");
const API = read("features/settings/api/download-transport.ts");
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


// The four below are source assertions, in the idiom of this file. They pin the shape of a
// fix rather than its behaviour, so they would not catch a rewrite that keeps the wording
// and breaks the rule; each names the failure it exists for.

test("the Hub's automatic fallback does not rewrite the install setting", () => {
  // TransportToggle drops a stored "xet" to "http" by itself when hf_xet is missing. Once the
  // setter also wrote install-wide, merely opening the Hub replaced everyone's choice for
  // good: repairing hf_xet later did not bring it back.
  assert.match(TOGGLE, /setMode\("http",\s*\{\s*persistInstall:\s*false\s*\}\)/);
  assert.match(PREFERENCE, /opts\.persistInstall === false/);
});

test("a download re-reads the install setting instead of trusting the cache", () => {
  // The point of an install-level setting is that another browser can change it. A value
  // cached for the life of the tab kept downloading on the old transport until a reload.
  assert.match(PREFERENCE, /hydrateInstallMode\(true\)/);
  assert.match(API, /opts\.refresh/);
});

test("install-wide writes are serialized", () => {
  // Two quick selections raced, and the earlier PUT landing last left the database on the
  // mode the user did not pick while this browser showed the one they did.
  assert.match(API, /writeQueue/);
  assert.match(API, /writeQueue\s*=\s*next\.catch/);
});

test("the copy stops promising a resume the install cannot do", () => {
  // huggingface_hub 1.18 made the HTTP writer process-unique, so an interrupted transfer is
  // refetched from zero. Someone picking HTTPS to keep their progress was told the opposite.
  assert.match(ROW, /useHttpPartialsResumable\(\)/);
  assert.match(ROW, /transportDescriptionNoResume/);
  assert.match(ROW, /httpsHintNoResume/);
  assert.match(EN, /transportDescriptionNoResume/);
  assert.match(EN, /httpsHintNoResume/);
});

test("the Xet-missing reason is the translated one", () => {
  // The backend has one reason for Xet being unavailable and it is English prose. Preferring
  // it made the translated key unreachable, so every non-English locale read English.
  assert.match(ROW, /hf_xet is not installed/);
  assert.match(ROW, /t\("settings\.general\.downloads\.xetMissing"\)/);
});
