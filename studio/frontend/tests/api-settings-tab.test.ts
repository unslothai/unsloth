// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const TAB = readSrc("features/settings/tabs/api-keys-tab.tsx");
const SUMMARY = readSrc("features/settings/components/remote-lan-summary.tsx");
const LAN_API = readSrc("features/settings/api/lan-access.ts");

test("the API tab shows Remote & LAN as a status line, not a second set of polling panels", () => {
  assert.doesNotMatch(TAB, /<RemoteAccessSection|<LanAccessSection/);
  assert.match(TAB, /<RemoteLanSummary \/>/);
  assert.doesNotMatch(SUMMARY, /setInterval/);
  assert.match(
    SUMMARY,
    /next\.state === "starting" \|\| next\.state === "stopping"\) \{\s*timer = window\.setTimeout/,
  );
  assert.match(SUMMARY, /setActiveTab\("remote-lan"\)/);
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
