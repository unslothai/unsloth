// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DEFAULT_TRANSPORT_MODE, TRANSPORT, pickTransportMode } = await import(
  "../src/features/hub/download-manager/constants.ts"
);

const ROW = readSrc("features/settings/components/download-transport-row.tsx");
const PREFERENCE = readSrc(
  "features/hub/download-manager/transport-preference.ts",
);
const GENERAL_TAB = readSrc("features/settings/tabs/general-tab.tsx");
const TOGGLE = readSrc("features/hub/catalog/transport-toggle.tsx");
const API = readSrc("features/settings/api/download-transport.ts");
const POLL_LOOP = readSrc("features/hub/download-manager/poll-loop.ts");
const SEARCH = readSrc("features/settings/settings-search.ts");
const GENERAL_TAB_SRC = readSrc("features/settings/tabs/general-tab.tsx");
const EN = readSrc("i18n/locales/en.ts");

test("this browser's own choice beats the install setting", () => {
  assert.equal(pickTransportMode("xet", "http"), TRANSPORT.XET);
  assert.equal(pickTransportMode("http", "auto"), TRANSPORT.HTTP);
  assert.equal(pickTransportMode("auto", "http"), TRANSPORT.AUTO);
});

test("with no choice of its own the install setting decides", () => {
  assert.equal(pickTransportMode(null, "auto"), TRANSPORT.AUTO);
  assert.equal(pickTransportMode(null, "xet"), TRANSPORT.XET);
  assert.equal(pickTransportMode(undefined, "http"), TRANSPORT.HTTP);
});

test("junk on either side leaves the default alone", () => {
  assert.equal(pickTransportMode("ftp", "torrent"), TRANSPORT.AUTO);
  assert.equal(pickTransportMode(null, null), DEFAULT_TRANSPORT_MODE);
  assert.equal(DEFAULT_TRANSPORT_MODE, TRANSPORT.AUTO);
});

test("a download waits for the install setting before picking a transport", () => {
  for (const source of [
    readSrc("features/hub/download-manager/poll-loop.ts"),
    readSrc("features/hub/download-manager/transport-conflict.ts"),
  ]) {
    assert.match(source, /await resolveTransportMode\(\)/);
    assert.ok(
      !/[^a-zA-Z]getTransportMode\(\)/.test(source),
      "a download start reads the preference without waiting for it",
    );
  }
});

test("choosing a transport saves it for the install too", () => {
  assert.match(PREFERENCE, /updateDownloadTransportSettings\(next\)/);
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
  assert.match(ROW, /xetAvailable === false/);
  assert.match(ROW, /autoResolvesTo/);
});

test("the copy explains the difference, not just the names", () => {
  const downloads = EN.slice(
    EN.indexOf("      downloads: {"),
    EN.indexOf("      uploads: {"),
  );
  assert.ok(downloads.length > 0, "the downloads copy moved");
  assert.match(downloads, /resumes/i);
  assert.match(downloads, /cancel/i);
  assert.match(downloads, /hf_xet/);
});

test("the Hub's automatic fallback is not stored at all", () => {
  assert.match(TOGGLE, /setMode\("http",\s*\{\s*persist:\s*false\s*\}\)/);
  assert.match(PREFERENCE, /opts\.persist === false/);
  const setter = PREFERENCE.slice(PREFERENCE.indexOf("const set = useCallback"));
  assert.ok(
    setter.indexOf("opts.persist === false") < setter.indexOf("localStorage.setItem"),
    "the persist opt-out must be checked before the local write",
  );
});

test("a download re-reads the install setting instead of trusting the cache", () => {
  assert.match(PREFERENCE, /hydrateInstallMode\(true\)/);
  assert.match(API, /opts\.refresh/);
});

test("install-wide writes are serialized", () => {
  assert.match(API, /writeQueue/);
  assert.match(API, /writeQueue\s*=\s*next\.catch/);
});

test("the copy stops promising a resume the install cannot do", () => {
  assert.match(ROW, /useHttpPartialsResumable\(\)/);
  assert.match(ROW, /transportDescriptionNoResume/);
  assert.match(ROW, /httpsHintNoResume/);
  assert.match(EN, /transportDescriptionNoResume/);
  assert.match(EN, /httpsHintNoResume/);
});

test("the Xet-missing reason is the translated one", () => {
  assert.match(ROW, /hf_xet is not installed/);
  assert.match(ROW, /t\("settings\.general\.downloads\.xetMissing"\)/);
});

test("a blocked localStorage still saves the setting for the install", () => {
  assert.match(PREFERENCE, /savedLocally/);
  assert.doesNotMatch(
    PREFERENCE,
    /catch \{\s*toast\.error\("Couldn't save the download transport preference\."\);\s*return;/,
  );
});

test("the untranslated health reason is not folded into a translated sentence", () => {
  assert.match(ROW, /statusReason/);
  assert.doesNotMatch(ROW, /autoCurrentlyReason/);
  assert.doesNotMatch(EN, /autoCurrentlyReason/);
});

test("a failed refresh keeps the install mode already loaded", () => {
  assert.match(PREFERENCE, /installMode === null && superseded \? superseded : installMode/);
});

test("a failed refresh waits for the hydration it overtook", () => {
  assert.match(PREFERENCE, /const superseded = refresh \? installModeInFlight : null;/);
});

test("adopting an existing job does not wait on the settings route", () => {
  assert.match(POLL_LOOP, /opts\.adopt\s*\n?\s*\? TRANSPORT\.HTTP/);
});

test("Xet cannot be chosen before its availability is known", () => {
  assert.match(ROW, /capabilityPending \|\| settings\?\.xetAvailable === false/);
  assert.match(ROW, /setCapabilityPending\(false\)/);
});

test("a failed install-wide write is reported, not just logged", () => {
  assert.match(PREFERENCE, /Saved for this browser, but not for this install\./);
});

test("the transport row is reachable from Settings search", () => {
  for (const key of [
    "settings.general.downloads.sectionTitle",
    "settings.general.downloads.transport",
    "settings.general.downloads.https",
    "settings.general.downloads.xet",
  ]) {
    assert.ok(SEARCH.includes(key), `search index is missing ${key}`);
  }
});

test("a refresh does not ride on a request that predates it", () => {
  assert.match(API, /inFlightIsRefresh/);
  assert.match(API, /!opts\.refresh \|\| inFlightIsRefresh/);
  assert.match(API, /request === latestRequest/);
});

test("the Hub toggle also waits to know whether Xet can run", () => {
  assert.match(TOGGLE, /const xetUnavailable = isLoading \|\| xetKnownUnavailable/);
});

test("each indexed option has somewhere for search to scroll to", () => {
  assert.match(ROW, /data-settings-label=\{t\(opt\.labelKey\)\}/);
});

test("the display only falls back once Xet is known unavailable", () => {
  assert.match(TOGGLE, /xetKnownUnavailable = capabilities\?\.xet\.available === false/);
  assert.match(TOGGLE, /mode === "xet" && xetKnownUnavailable/);
  assert.match(TOGGLE, /isLoading \|\| xetKnownUnavailable/);
});

test("a refresh is not swallowed by the hydration wrapper", () => {
  assert.match(PREFERENCE, /installModeInFlightIsRefresh/);
  assert.match(
    PREFERENCE,
    /installModeInFlight && \(!refresh \|\| installModeInFlightIsRefresh\)/,
  );
});

test("resetting local preferences clears the transport override", () => {
  assert.match(PREFERENCE, /export const TRANSPORT_MODE_STORAGE_KEY/);
  assert.match(GENERAL_TAB_SRC, /TRANSPORT_MODE_STORAGE_KEY/);
});

test("a completed write outranks a read issued before it", () => {
  assert.match(API, /latestRequest \+= 1;/);
});

test("the settings row re-reads the install setting when it opens", () => {
  assert.match(ROW, /loadDownloadTransportSettings\(\{ refresh: true \}\)/);
});

test("a superseded read answers with the current value, not its own", () => {
  assert.match(API, /return cachedTransport \?\? settings;/);
});

test("the transport controls mount on a refreshed install mode", () => {
  assert.match(PREFERENCE, /hydrateInstallMode\(true\)\.then\(\(\) => setMode/);
});
