// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const TAB = readSrc("features/settings/tabs/api-keys-tab.tsx");
const LAN_API = readSrc("features/settings/api/lan-access.ts");

test("the API tab leaves Remote & LAN to its own tab, the next one down", () => {
  assert.doesNotMatch(
    TAB,
    /<RemoteAccessSection|<LanAccessSection|<RemoteLanSummary/,
  );
});

test("the API tab still publishes a tunnel URL that comes up after it opens", () => {
  const hook = readSrc("features/settings/hooks/use-publish-access-urls.ts");
  assert.match(TAB, /usePublishAccessUrls\(isOwner\);/);
  assert.match(
    hook,
    /usePlatformStore\.setState\(\{ cloudflareUrl: next\.url \}\)/,
  );
  assert.match(
    hook,
    /next\.state === "starting" \|\| next\.state === "stopping"/,
  );
  assert.match(
    hook,
    /window\.setTimeout\(readRemote, remoteAccessPollDelay\(next\)\)/,
  );
  assert.match(hook, /publishLanUrls\(next\)/);
});

test("usage examples sit right under the keys, where a new key lands", () => {
  const examples = TAB.indexOf("<UsageExamples");
  assert.ok(examples > TAB.indexOf("<CreateKeyForm"));
  assert.ok(examples < TAB.indexOf("<MonitorLink />"));
  assert.ok(examples < TAB.indexOf("<KeylessApiAccessSection"));
});

test("the remote and LAN polls skip their request while the tab is hidden", () => {
  for (const path of [
    "features/settings/components/lan-access-section.tsx",
    "features/settings/components/remote-access-section.tsx",
  ]) {
    assert.match(
      readSrc(path),
      /if \(document\.hidden\) \{\s*schedule\([^)]*\);\s*return;\s*\}/,
      path,
    );
  }
});

test("LAN URLs publish only when they change", () => {
  assert.match(
    LAN_API,
    /next\.every\(\(url, i\) => url === current\[i\]\)[\s\S]{0,40}return;/,
  );
  assert.doesNotMatch(
    readSrc("features/settings/components/lan-access-section.tsx"),
    /setState\(\{ lanUrls/,
  );
  assert.doesNotMatch(
    readSrc("features/api-monitor/api-monitor-page.tsx"),
    /setState\(\{ lanUrls/,
  );
});

test("the API page marks one served row as the primary, found by checkpoint first", () => {
  const page = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.match(page, /active: index === activeIndex,/);
  assert.match(
    page,
    /\(model\.checkpoint \?\? model\.id\) === residentCheckpoint/,
  );
  assert.match(page, /\(model\.active && !canEditSettings\)/);
});

test("the keyless note shows beside a public address too, since LAN callers stay keyless", () => {
  const lan = readSrc("features/settings/components/lan-access-section.tsx");
  assert.doesNotMatch(
    lan,
    /keylessLanEligible &&\s*status\.publicUrls\.length === 0/,
  );
  assert.equal(lan.match(/\{keylessNote\}/g)?.length, 2);
});
