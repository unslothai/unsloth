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
  assert.ok(examples < TAB.indexOf("<KeylessApiAccessLink"));
});

test("jump links above the token form reach the tab's APIs from any page, which sit images first, then audio, then decisions", () => {
  const links = readSrc("features/settings/components/api-jump-links.tsx");
  const examples = readSrc("features/settings/components/usage-examples.tsx");
  for (const target of ["api-keys-media-api", "api-keys-audio-api", "api-keys-decision-api", "api-keys-agents", "api-keys-mcp"]) {
    assert.match(links, new RegExp(`target: "${target}"`));
  }
  assert.match(links, /openDialog\("api-keys", \{ scrollTarget: link\.target \}\)/);
  assert.match(examples, /scrollTarget !== "api-keys-agents" && scrollTarget !== "api-keys-mcp"/);
  assert.match(examples, /consumeScrollTarget\(target\)/);
  assert.match(examples, /<div ref=\{agentsRef\} className=/);
  assert.match(examples, /<div ref=\{mcpRef\}>\{agentsExtra\}<\/div>/);
  assert.match(examples, /if \(agentsOnly\) return;\s*if \(scrollTarget !== "api-keys-agents"/);
  assert.doesNotMatch(examples, /getElementById/);
  for (const [file, target] of [
    ["media-api-examples.tsx", "api-keys-media-api"],
    ["audio-api-examples.tsx", "api-keys-audio-api"],
    ["decision-api-section.tsx", "api-keys-decision-api"],
  ]) {
    assert.match(readSrc(`features/settings/components/${file}`), new RegExp(`if \\(!showHeading \\|\\| scrollTarget !== "${target}"`), file);
  }
  assert.ok(TAB.indexOf("<ApiJumpLinks") > 0);
  assert.ok(TAB.indexOf("<ApiJumpLinks") < TAB.indexOf("<CreateKeyForm"));
  assert.match(TAB, /<h2 className="settings-heading text-base font-semibold font-heading">\s*\{t\("settings\.apiKeys\.accessTokens"\)\}\s*<\/h2>\s*<AnimatePresence/);
  assert.equal(TAB.split('t("settings.apiKeys.accessTokens")').length - 1, 1);
  const monitor = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.doesNotMatch(monitor, /<ApiJumpLinks/);
  for (const part of [
    /<MediaApiExamples useTunnel=\{useTunnel\} showHeading=\{false\} \/>/,
    /<AudioApiExamples useTunnel=\{useTunnel\} showHeading=\{false\} \/>/,
    /<DecisionApiSection showHeading=\{false\} \/>/,
    /<UsageExamples[^>]*agentsOnly=\{true\}/,
  ]) {
    assert.match(monitor, part);
  }
  assert.match(monitor, /\.\.\.\(isOwner\s*\? \[\{ id: "decision" as const/);
  assert.doesNotMatch(monitor, /McpAccessSection|id: "mcp"/);
  assert.match(monitor, /font-heading text-ui-15/);
  assert.match(examples, /\{agentsOnly \? null : \(/);
  for (const icon of ["MessageCircleIcon", "Image01Icon", "AiSpeechIcon", "TaskDone01Icon", "BotIcon"]) {
    assert.match(monitor, new RegExp(`label: "[^"]+", icon: ${icon} \\}`));
  }
  assert.ok(TAB.indexOf("<MediaApiExamples") < TAB.indexOf("<AudioApiExamples"));
  assert.ok(TAB.indexOf("<AudioApiExamples") < TAB.indexOf("<DecisionApiSection"));
  const decision = readSrc("features/settings/components/decision-api-section.tsx");
  assert.match(
    decision,
    /<h2 className="settings-heading flex items-center gap-2 text-base font-semibold font-heading">\s*<HugeiconsIcon\s+icon=\{TaskDone01Icon\}/,
  );
});

test("keyless API access is edited on Remote & LAN; the API tab links there and reads it for the examples", () => {
  const link = readSrc("features/settings/components/keyless-api-access-link.tsx");
  assert.match(link, /openDialog\("remote-lan", \{ scrollTarget: "remote-lan-keyless" \}\)/);
  const remoteLan = readSrc("features/settings/tabs/remote-lan-tab.tsx");
  assert.match(remoteLan, /scrollTarget !== "remote-lan-keyless"/);
  assert.match(remoteLan, /<div ref=\{keylessRef\}>\s*<KeylessApiAccessSection \/>/);
  assert.match(TAB, /loadKeylessApiAccess\(\)/);
  assert.match(TAB, /scope: next\.suspendedByAccounts \? "off" : next\.scope/);
  assert.doesNotMatch(TAB, /<KeylessApiAccessSection/);
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
