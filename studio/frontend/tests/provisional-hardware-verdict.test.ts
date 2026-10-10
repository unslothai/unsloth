// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /api/health reports chat_only: true before detection; storing it sends a GPU host to /chat.

import assert from "node:assert/strict";
import test from "node:test";

import {
  readSrc,
  readSrcAsync,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();

const { isDetectionDeferred, isProvisionalVerdict, resolveVerdict, videoNavHint } = await import(
  "../src/config/hardware-verdict.ts"
);

const APP_SIDEBAR = readSrc("components/app-sidebar.tsx");
const ENV = readSrc("config/env.ts");
const USE_HARDWARE_INFO = readSrc("hooks/use-hardware-info.ts");

const GPU_HOST = { chatOnly: false, chatOnlyReason: null, chatOnlyDetail: null };
const MAC_DEFAULT = { chatOnly: true, chatOnlyReason: null, chatOnlyDetail: null };

test("a detecting reply is provisional, a settled one is not", () => {
  assert.equal(isProvisionalVerdict({ hardware_detecting: true }), true);
  assert.equal(isProvisionalVerdict({ hardware_detecting: false }), false);
  assert.equal(
    isProvisionalVerdict({ chat_only: false }),
    false,
    "a reply with no hardware_detecting at all is a measured one",
  );
});

test("a provisional reply does not send a GPU host to chat-only", () => {
  const resolved = resolveVerdict(
    { chat_only: true, hardware_detecting: true },
    GPU_HOST,
  );
  assert.equal(
    resolved.chatOnly,
    false,
    "the provisional chat_only was stored; beforeLoad would redirect a GPU host to /chat",
  );
});

test("a provisional reply does not clear a reason the UI is explaining", () => {
  const resolved = resolveVerdict(
    { chat_only: true, hardware_detecting: true },
    { chatOnly: true, chatOnlyReason: "mlx_unavailable", chatOnlyDetail: null },
  );
  assert.equal(
    resolved.chatOnlyReason,
    "mlx_unavailable",
    "the sidebar recovery poll only runs while it reads mlx_unavailable",
  );
});

test("a measured chat-only verdict is still honoured", () => {
  const resolved = resolveVerdict(
    { chat_only: true, chat_only_reason: "mlx_unavailable" },
    GPU_HOST,
  );
  assert.equal(resolved.chatOnly, true, "a real chat-only host was let into Train");
  assert.equal(resolved.chatOnlyReason, "mlx_unavailable");
});

test("a measured GPU verdict clears a chat-only default", () => {
  const resolved = resolveVerdict(
    { chat_only: false, chat_only_reason: null },
    MAC_DEFAULT,
  );
  assert.equal(
    resolved.chatOnly,
    false,
    "keeping the previous value has to stop once a measurement arrives",
  );
});

test("a measured reply with no chat_only field is not chat-only", () => {
  assert.equal(resolveVerdict({}, MAC_DEFAULT).chatOnly, false);
});

test("a deferred verdict is provisional but must not be waited on", () => {
  const deferred = { chat_only: true, hardware_detecting: true, hardware_detection_deferred: true };
  assert.equal(
    isProvisionalVerdict(deferred),
    true,
    "a deferred reply is still not a measurement, so it must not be stored",
  );
  assert.equal(
    isDetectionDeferred(deferred),
    true,
    "the kill switch stops anything settling, so the re-read loop must give up",
  );
});

test("a deferred verdict falls back to the backend's conservative default", () => {
  // With the kill switch on nothing settles, so keeping the old value strands a CPU host.
  const deferred = {
    chat_only: true,
    hardware_detecting: true,
    hardware_detection_deferred: true,
  };
  assert.equal(
    resolveVerdict(deferred, { chatOnly: false, chatOnlyReason: null, chatOnlyDetail: null }).chatOnly,
    true,
    "a never-settling reply left the optimistic platform default in place",
  );
});

test("a deferred verdict does not clear a reason the UI is explaining", () => {
  const deferred = {
    chat_only: true,
    hardware_detecting: true,
    hardware_detection_deferred: true,
  };
  assert.equal(
    resolveVerdict(deferred, { chatOnly: true, chatOnlyReason: "mlx_unavailable", chatOnlyDetail: null })
      .chatOnlyReason,
    "mlx_unavailable",
    "the sidebar recovery poll only runs while it reads mlx_unavailable",
  );
});

test("an ordinary provisional reply is still not treated as deferred", () => {
  // Taking a warm-window chat_only would send a GPU host to /chat.
  assert.equal(
    resolveVerdict({ chat_only: true, hardware_detecting: true }, GPU_HOST).chatOnly,
    false,
  );
});

test("an actively detecting reply is not deferred", () => {
  assert.equal(
    isDetectionDeferred({ chat_only: true, hardware_detecting: true }),
    false,
    "an ordinary warm-window reply must still be waited on",
  );
});

// Asserted on source: env.ts is not importable outside vite.
test("the bounded hardware wait is spent at most once per page load", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    ENV,
    /let hardwareWaitSpent = false/,
    "no once-per-load latch: every navigation can repeat the full wait",
  );
  assert.match(
    ENV,
    /const deadline = spendWait \? Date\.now\(\) \+ HARDWARE_DETECT_WAIT_MS : 0/,
    "the guard does not zero the deadline, so the wait is not actually skipped",
  );
});

// /api/health reports device_type to authed callers only, so an unauthed poll never settles.
test("an unauthenticated read never spends the detection window", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    ENV,
    /const spendWait = Boolean\(token\) && !hardwareWaitSpent/,
    "the wait is not gated on having a token, so /login blocks on hardware detection",
  );
});

// main.tsx fires an unawaited fetchDeviceType while __root.tsx awaits its own.
test("the latch is claimed after the wait, not during it", async () => {
  const { readFile } = await import("node:fs/promises");
  const loopStart = ENV.indexOf("while (res.ok && Date.now() < deadline)");
  // Anchor on the loop's closing brace, not the latch, or the slice moves with the regression.
  const loopEnd = ENV.indexOf("\n    }\n", loopStart);
  assert.ok(loopStart > 0 && loopEnd > loopStart, "the wait loop moved");
  assert.ok(
    !ENV.slice(loopStart, loopEnd).includes("hardwareWaitSpent = true"),
    "the latch is claimed inside the loop, so a concurrent caller skips an unfinished window",
  );
  assert.match(
    ENV.slice(loopEnd),
    /if \(spendWait[^)]*\) hardwareWaitSpent = true;/,
    "the latch is never claimed after the wait, so the window is never spent",
  );
});

// Falling back to the browser platform mislabels WSL/SSH remote hosts as local.
test("a provisional forced refresh keeps the server-reported platform", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    ENV,
    /const keepPlatform =\s*\n?\s*data\.device_type === undefined && previous\.fetched/,
    "no guard: a provisional forced refresh overwrites the authoritative platform",
  );
  assert.match(
    ENV,
    /keepPlatform \? previous\.deviceType : detectLocalPlatform\(\)/,
    "the guard does not actually keep the stored platform",
  );
  assert.match(
    ENV,
    /fetched: data\.device_type !== undefined \|\| keepPlatform/,
    "fetched drops to false, so the authoritative platform is not treated as held",
  );
});

// Under the kill switch only a first-use operation detects, so the sidebar must poll.
test("a deferred verdict is recorded so the sidebar can poll out of it", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    ENV,
    /detectionDeferred: isDetectionDeferred\(data\)/,
    "the store never records that the verdict came from a deferred reply",
  );
  assert.match(
    APP_SIDEBAR,
    /chatOnlyReason !== "mlx_unavailable" && !detectionDeferred/,
    "the recovery poll still ignores a deferred verdict, so it never recovers",
  );
});

// /api/health answers a rejected token with the unauthed body, so a stale token holds /login.
test("a rejected token stops the wait instead of polling it out", async () => {
  const { readFile } = await import("node:fs/promises");
  const loopStart = ENV.indexOf("while (res.ok && Date.now() < deadline)");
  const loopEnd = ENV.indexOf("\n    }\n", loopStart);
  assert.ok(loopStart > 0 && loopEnd > loopStart, "the wait loop moved");
  const loop = ENV.slice(loopStart, loopEnd);
  assert.ok(
    /if \(peek\.version === undefined\)\s*\{?[^}]*break;/.test(loop),
    "the loop keeps polling a reply with no authed-only field, so an expired token " +
      "waits out the full window on /login",
  );
  assert.match(
    loop,
    /version\?: string;/,
    "the peek type no longer reads the authed-only field it breaks on",
  );
});

// Spending the latch on a refused token leaves the route guard on defaults until refresh.
test("a rejected token does not consume the once-per-load window", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    ENV,
    /if \(spendWait && !tokenRejected\) hardwareWaitSpent = true;/,
    "the latch is claimed even when the break came from a rejected token, so the " +
      "first accepted token in this page load skips its window",
  );
  assert.match(
    ENV,
    /tokenRejected = true;/,
    "nothing records that the break was caused by a rejected token",
  );
});

// chatOnly is seeded from the user agent, so Macs read true before /api/health answers.
test("the store exposes an unknown state, not just chat-only", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    ENV,
    /capabilitiesUnknown: \(\) => boolean;/,
    "PlatformState has no unknown state, so a caller can only read the guess",
  );
  assert.match(
    ENV,
    /return !state\.fetched && !state\.detectionDeferred;/,
    "the selector is not derived from `fetched`, the flag that already means " +
      "'a server-reported verdict is stored'",
  );
});

// Under the torch-warm kill switch `fetched` never flips, so unknown would spin forever.
test("a deferred verdict counts as settled, not as still checking", async () => {
  const { readFile } = await import("node:fs/promises");
  const selector = /capabilitiesUnknown: \(\) => \{([\s\S]*?)\n  \},/.exec(ENV);
  assert.ok(selector, "capabilitiesUnknown is no longer a block the deferred case can live in");
  assert.match(
    selector[1],
    /detectionDeferred/,
    "a deferred reply leaves the tabs spinning with nothing left to wait for",
  );
});

test("the sidebar gates Train and Video on a measured verdict", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    APP_SIDEBAR,
    /const chatOnlyMeasured = chatOnly && !capabilitiesUnknown;/,
    "the rows still read chatOnly directly, so the UA guess disables them",
  );
  // chatOnly is an object entry ending in a comma; the class excludes chatOnlyMeasured.
  for (const pattern of [/disabled: chatOnly[,;\s]/, /if \(chatOnly\) return;/]) {
    assert.ok(
      !pattern.test(APP_SIDEBAR),
      `${pattern} still reads the unmeasured verdict`,
    );
  }
  assert.equal(
    APP_SIDEBAR.match(/pending: capabilitiesUnknown,/g)?.length,
    2,
    "Train and Video do not both mark themselves pending while the verdict is out",
  );
});

// beforeLoad redirects are one-way, so a pre-measurement bounce strands the host on /chat.
test("the route guard waits out an unknown verdict on Train and Video", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readSrcAsync("app/routes/__root.tsx");
  const guard = /const SELF_GATED_WHILE_UNKNOWN = \[([^\]]*)\]/.exec(src);
  assert.ok(guard, "no list of paths that wait the verdict out");
  for (const path of ["/studio", "/video"]) {
    assert.ok(guard[1].includes(`"${path}"`), `${path} is still redirected on the guess`);
  }
  assert.match(
    src,
    /!\(unmeasured && waitsOutUnknownVerdict\(location\.pathname\)\)/,
    "the redirect does not consult the unknown state",
  );
  // Child routes count too, or /studio/anything bounces while /studio does not.
  assert.match(
    src,
    /pathname === base \|\| pathname\.startsWith\(`\$\{base\}\/`\)/,
    "only the exact path waits the verdict out",
  );
});

test("the Video page gates on the backend's own capability answer", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readSrcAsync("features/video/video-page.tsx");
  assert.match(
    src,
    /hardware\.videoSupported === false/,
    "the page does not read the backend's video verdict",
  );
  assert.match(
    src,
    /if \(!hardware\.loaded\)/,
    "the page renders its generator before the verdict has landed",
  );
  // Video does not use MLX, so it must not wait on the training verdict during MLX self-heal.
  assert.ok(
    !/usePlatformStore/.test(src),
    "the Video gate waits on the training verdict again",
  );
  // An older backend sends null; only an explicit false may hide the generator.
  assert.ok(
    !/videoSupported !== true/.test(src),
    "an older backend's missing field would hide the page it has always served",
  );
});

test("the Video hint names the reason the host has no video device", () => {
  assert.equal(videoNavHint(false, null), undefined, "a capable host is disabled");
  assert.equal(videoNavHint(false, "no_gpu"), undefined, "an unmeasured host is disabled");
  assert.equal(videoNavHint(false, "intel_mac"), undefined, "an unmeasured host is disabled");
  assert.equal(
    videoNavHint(true, "no_gpu"),
    "Video generation needs an NVIDIA or AMD GPU.",
    "a GPU-less host is not told why",
  );
  assert.match(
    videoNavHint(true, "intel_mac") ?? "",
    /Apple Silicon/,
    "an Intel Mac is not told what video actually needs",
  );
  // torch's only macOS backend is MPS, which needs Apple Silicon, so no GPU is a fix.
  assert.ok(
    !/gpu|graphics|coming soon/i.test(videoNavHint(true, "intel_mac") ?? ""),
    "an Intel Mac is pointed at graphics hardware that cannot run video",
  );
});

// Video runs on Metal, not MLX, so a broken MLX stack must not disable the row.
test("a chat-only Apple Silicon host keeps Video navigable", () => {
  assert.equal(
    videoNavHint(true, "mlx_unavailable"),
    undefined,
    "a repairable MLX stack disables video, which does not use MLX",
  );
  assert.equal(videoNavHint(true, "detection_failed"), undefined);
  assert.equal(videoNavHint(true, null), undefined);
});

test("the Video row is disabled exactly when the hint has something to say", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    APP_SIDEBAR,
    /const videoDisabledHint = videoNavHint\(chatOnlyMeasured, chatOnlyReason\)/,
    "the Video hint no longer comes from the shared derivation",
  );
  assert.match(
    APP_SIDEBAR,
    /const videoDisabled = videoDisabledHint !== undefined/,
    "the Video row's disabled state is no longer derived from its hint",
  );
  // The train row keeps `disabled: chatOnlyMeasured`, so match inside the video row only.
  const videoRow = APP_SIDEBAR.slice(APP_SIDEBAR.indexOf("    video: {"));
  const videoBody = videoRow.slice(0, videoRow.indexOf("},"));
  assert.match(
    videoBody,
    /disabled: videoDisabled,/,
    "the Video row is gated on the training verdict again",
  );
  assert.match(
    videoBody,
    /tooltip: videoDisabledHint,/,
    "the Video row is disabled with no reason shown",
  );
  assert.ok(!/coming soon/.test(APP_SIDEBAR), "the Video row still promises macOS support that has shipped");
});

// A failed probe used to stay unloaded forever, spinning the Video gate.
test("a failed hardware probe is retried, not left unloaded", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    USE_HARDWARE_INFO,
    /if \(!cancelled && !hw\.loaded\) retry = setTimeout\(load, RETRY_MS\);/,
    "nothing re-probes after a failed read",
  );
  assert.match(
    USE_HARDWARE_INFO,
    /if \(retry !== undefined\) clearTimeout\(retry\);/,
    "the retry outlives the component that scheduled it",
  );
});

// A probe resolving between render and subscribe would otherwise never call setInfo.
test("a cache filled between render and subscribe still reaches the component", async () => {
  const { readFile } = await import("node:fs/promises");
  assert.match(
    USE_HARDWARE_INFO,
    /if \(cached\) listener\(cached\);\s*\n\s*else load\(\);/,
    "a component that missed the notify has no path to the cache it skipped loading for",
  );
  // load() reads !loaded as failure, so a superseded 200 must not count as one.
  assert.ok(
    !/return cached \?\? DEFAULT;/.test(USE_HARDWARE_INFO),
    "a superseded but successful read still resolves as an unloaded DEFAULT",
  );
});
