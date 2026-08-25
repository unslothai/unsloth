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
const POLL_LOOP = read("features/hub/download-manager/poll-loop.ts");
const SEARCH = read("features/settings/settings-search.ts");
const GENERAL_TAB_SRC = read("features/settings/tabs/general-tab.tsx");
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

test("the Hub's automatic fallback is not stored at all", () => {
  // TransportToggle drops a stored "xet" to "http" by itself when hf_xet is missing. Writing
  // that install-wide replaced everyone's choice; writing it locally was no better, since a
  // local value outranks the install one, so repairing hf_xet later never brought Xet back.
  assert.match(TOGGLE, /setMode\("http",\s*\{\s*persist:\s*false\s*\}\)/);
  assert.match(PREFERENCE, /opts\.persist === false/);
  // Reflected and returned BEFORE either write.
  const setter = PREFERENCE.slice(PREFERENCE.indexOf("const set = useCallback"));
  assert.ok(
    setter.indexOf("opts.persist === false") < setter.indexOf("localStorage.setItem"),
    "the persist opt-out must be checked before the local write",
  );
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

test("a blocked localStorage still saves the setting for the install", () => {
  // Private mode, storage disabled or over quota used to return before the server write,
  // so those browsers could not change the transport at all even with a healthy backend.
  assert.match(PREFERENCE, /savedLocally/);
  assert.doesNotMatch(
    PREFERENCE,
    /catch \{\s*toast\.error\("Couldn't save the download transport preference\."\);\s*return;/,
  );
});

test("the untranslated health reason is not folded into a translated sentence", () => {
  // settings.autoReason is free-form English from the Xet health check. Interpolating it
  // gave every non-English locale half a sentence in its own language and half in English,
  // so it gets its own line and the key that interpolated it is gone.
  assert.match(ROW, /statusReason/);
  assert.doesNotMatch(ROW, /autoCurrentlyReason/);
  assert.doesNotMatch(EN, /autoCurrentlyReason/);
});

test("a failed refresh keeps the install mode already loaded", () => {
  // Falling back to null sent the next download to Auto even though this browser still knew
  // the install's choice, so a blip on the settings route silently changed transport.
  assert.match(PREFERENCE, /\.catch\(\(\) => installMode\)/);
});

test("adopting an existing job does not wait on the settings route", () => {
  // The adopt branch ignores requestedMode, and suspending there let the two concurrent
  // adoptJob callers replace each other's runtime, leaving duplicate poll timers behind.
  assert.match(POLL_LOOP, /opts\.adopt\s*\n?\s*\? TRANSPORT\.HTTP/);
});

test("Xet cannot be chosen before its availability is known", () => {
  // Clicking it in that window stored a Xet preference, locally and install-wide, that every
  // later download silently ignored on a machine without hf_xet.
  assert.match(ROW, /capabilityPending \|\| settings\?\.xetAvailable === false/);
  // But a load that FAILED must not disable it for good, so pending is its own state.
  assert.match(ROW, /setCapabilityPending\(false\)/);
});

test("a failed install-wide write is reported, not just logged", () => {
  // The row calls this setting install-wide. A browser that kept the choice locally while the
  // install did not is a mismatch the user cannot otherwise see, and reload hides it further.
  assert.match(PREFERENCE, /Saved for this browser, but not for this install\./);
});

test("the transport row is reachable from Settings search", () => {
  // The whole reason it moved here is that nobody found it on the Hub. Leaving it out of the
  // index means searching for "transport", "Xet" or "HTTPS" still finds nothing.
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
  // The hydration GET may already have been answered with the old mode while still pending
  // here, so sharing it would hand a download exactly the value the refresh went to replace.
  assert.match(API, /inFlightIsRefresh/);
  assert.match(API, /!opts\.refresh \|\| inFlightIsRefresh/);
  // And the older of two overlapping responses must not land last and re-cache the old value.
  assert.match(API, /request === latestRequest/);
});

test("the Hub toggle also waits to know whether Xet can run", () => {
  // Same window as the settings row: clicking Xet before the capability lands stored it
  // locally and install-wide on a machine that cannot run it.
  assert.match(TOGGLE, /const xetUnavailable = isLoading \|\| xetKnownUnavailable/);
});

test("each indexed option has somewhere for search to scroll to", () => {
  // An indexed label with no data-settings-label produces a result that opens General and
  // then fails to find anything.
  assert.match(ROW, /data-settings-label=\{t\(opt\.labelKey\)\}/);
});

test("the display only falls back once Xet is known unavailable", () => {
  // Disabling on unknown and falling back on unknown are different rules. Conflating them
  // showed HTTP for a stored Xet during the capability check, with nothing to restore it if
  // Xet turned out to be fine.
  assert.match(TOGGLE, /xetKnownUnavailable = capabilities\?\.xet\.available === false/);
  assert.match(TOGGLE, /mode === "xet" && xetKnownUnavailable/);
  assert.match(TOGGLE, /isLoading \|\| xetKnownUnavailable/);
});

test("a refresh is not swallowed by the hydration wrapper", () => {
  // The API layer decides what may share a request; memoizing above it hid the refresh flag.
  assert.match(PREFERENCE, /installModeInFlightIsRefresh/);
  assert.match(
    PREFERENCE,
    /installModeInFlight && \(!refresh \|\| installModeInFlightIsRefresh\)/,
  );
});

test("resetting local preferences clears the transport override", () => {
  // It outranks the install-wide setting, so leaving it behind means the reset browser keeps
  // ignoring transport changes made anywhere else on the install.
  assert.match(PREFERENCE, /export const TRANSPORT_MODE_STORAGE_KEY/);
  assert.match(GENERAL_TAB_SRC, /TRANSPORT_MODE_STORAGE_KEY/);
});

test("a completed write outranks a read issued before it", () => {
  // Ordering GETs against each other is not enough: a GET taken before the PUT could still
  // land after it and republish the mode the user had just replaced.
  assert.match(API, /latestRequest \+= 1;/);
});

test("the settings row re-reads the install setting when it opens", () => {
  // Otherwise reopening Settings shows whatever is cached: a mode changed in another browser,
  // and an Auto verdict from before RAM pressure or a recorded Xet failure.
  assert.match(ROW, /loadDownloadTransportSettings\(\{ refresh: true \}\)/);
});
