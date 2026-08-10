// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /api/health answers before the backend has measured the host: until the background torch
// import lands it reports its pre-detection default, chat_only: true, with
// hardware_detecting set and no device_type. __root.tsx's beforeLoad awaits fetchDeviceType
// then redirects on isChatOnly(), so storing that sends a GPU host's first load to /chat.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isDetectionDeferred, isProvisionalVerdict, resolveVerdict } = await import(
  "../src/config/hardware-verdict.ts"
);

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
  // Nothing settles while the kill switch is on, so keeping the previous value would leave
  // the browser-platform default (chatOnly false off macOS) up all session on a CPU host.
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
  // The conservative fallback stays scoped to the kill switch: a warm-window reply
  // settles in ~1-2s, and taking its chat_only would send a GPU host to /chat.
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

// The bounded re-read in config/env.ts is spent at most once per page load: a slow host
// leaves the reply provisional and `fetched` false, so an unguarded loop repeats the full
// wait on every navigation. Asserted on source: env.ts is not importable outside vite.
test("the bounded hardware wait is spent at most once per page load", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /let hardwareWaitSpent = false/,
    "no once-per-load latch: every navigation can repeat the full wait",
  );
  assert.match(
    src,
    /const deadline = spendWait \? Date\.now\(\) \+ HARDWARE_DETECT_WAIT_MS : 0/,
    "the guard does not zero the deadline, so the wait is not actually skipped",
  );
});

// beforeLoad awaits fetchDeviceType on every route, /login included, and /api/health
// reports device_type to authed callers only. An unauthenticated poll can never turn
// provisional into measured, so it only holds the login form behind the torch import.
test("an unauthenticated read never spends the detection window", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /const spendWait = Boolean\(token\) && !hardwareWaitSpent/,
    "the wait is not gated on having a token, so /login blocks on hardware detection",
  );
});

// main.tsx fires an unawaited fetchDeviceType while __root.tsx awaits its own. Claiming
// the latch inside the loop let the unawaited one consume a window it had not finished,
// leaving the awaited caller on the local default and redirecting a GPU host to /chat.
test("the latch is claimed after the wait, not during it", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  const loopStart = src.indexOf("while (res.ok && Date.now() < deadline)");
  // Anchor on the loop's own closing brace, NOT on the latch. Searching for the latch
  // made this assertion unfalsifiable: String.prototype.search returns the FIRST match,
  // so moving the latch inside the loop moved loopEnd with it and the slice ended just
  // short of the claim either way. The test then passed against the exact regression its
  // header describes.
  const loopEnd = src.indexOf("\n    }\n", loopStart);
  assert.ok(loopStart > 0 && loopEnd > loopStart, "the wait loop moved");
  assert.ok(
    !src.slice(loopStart, loopEnd).includes("hardwareWaitSpent = true"),
    "the latch is claimed inside the loop, so a concurrent caller skips an unfinished window",
  );
  // And it must still be claimed somewhere after the loop, or deleting it outright
  // would satisfy the check above.
  assert.match(
    src.slice(loopEnd),
    /if \(spendWait[^)]*\) hardwareWaitSpent = true;/,
    "the latch is never claimed after the wait, so the window is never spent",
  );
});

// A provisional reply omits device_type. A forced refresh in that window must not fall
// back to the browser platform: on WSL, SSH or any remote session that relabels the host
// as local, changing model filtering, paths and install commands.
test("a provisional forced refresh keeps the server-reported platform", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /const keepPlatform =\s*\n?\s*data\.device_type === undefined && previous\.fetched/,
    "no guard: a provisional forced refresh overwrites the authoritative platform",
  );
  assert.match(
    src,
    /keepPlatform \? previous\.deviceType : detectLocalPlatform\(\)/,
    "the guard does not actually keep the stored platform",
  );
  assert.match(
    src,
    /fetched: data\.device_type !== undefined \|\| keepPlatform/,
    "fetched drops to false, so the authoritative platform is not treated as held",
  );
});

// With the kill switch on nothing settles through health, so only a first-use operation
// detects. The sidebar's recovery poll was gated on mlx_unavailable alone, so a GPU host
// kept the conservative deferred verdict and stayed chat-only until a hard refresh.
test("a deferred verdict is recorded so the sidebar can poll out of it", async () => {
  const { readFile } = await import("node:fs/promises");
  const env = await readFile(new URL("../src/config/env.ts", import.meta.url), "utf8");
  const sidebar = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    env,
    /detectionDeferred: isDetectionDeferred\(data\)/,
    "the store never records that the verdict came from a deferred reply",
  );
  assert.match(
    sidebar,
    /chatOnlyReason !== "mlx_unavailable" && !detectionDeferred/,
    "the recovery poll still ignores a deferred verdict, so it never recovers",
  );
});

// A token in localStorage is not proof of an accepted one. /api/health catches the auth
// failure and answers with the unauthenticated body, which never carries device_type, so
// a stale token spends the whole window and holds /login on a cold boot.
test("a rejected token stops the wait instead of polling it out", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  const loopStart = src.indexOf("while (res.ok && Date.now() < deadline)");
  // Same brace anchor as above rather than the latch pattern, so the slice cannot move
  // with the code it is meant to be measuring.
  const loopEnd = src.indexOf("\n    }\n", loopStart);
  assert.ok(loopStart > 0 && loopEnd > loopStart, "the wait loop moved");
  const loop = src.slice(loopStart, loopEnd);
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

// Breaking early on a rejected token must not consume the once-per-page-load window. A
// user who signs in before detection settles gets the first authenticated read; spending
// the latch on a refused token leaves the route guard on local defaults until a refresh.
test("a rejected token does not consume the once-per-load window", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /if \(spendWait && !tokenRejected\) hardwareWaitSpent = true;/,
    "the latch is claimed even when the break came from a rejected token, so the " +
      "first accepted token in this page load skips its window",
  );
  assert.match(
    src,
    /tokenRejected = true;/,
    "nothing records that the break was caused by a rejected token",
  );
});

// The same guess, one layer up. `chatOnly` is seeded from the user agent, so on every Mac it
// reads true from first paint, before /api/health has said anything. Train and Video rendered
// straight off it, so both tabs blacked out on launch and only came back once the backend
// answered -- which PR #7607's lazy detection can stretch to minutes.
test("the store exposes an unknown state, not just chat-only", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /capabilitiesUnknown: \(\) => boolean;/,
    "PlatformState has no unknown state, so a caller can only read the guess",
  );
  assert.match(
    src,
    /return !state\.fetched && !state\.detectionDeferred;/,
    "the selector is not derived from `fetched`, the flag that already means " +
      "'a server-reported verdict is stored'",
  );
});

// The torch-warm kill switch (UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1) settles nothing until a
// hardware-dependent operation runs, and a deferred reply carries no device_type, so `fetched`
// never flips. Calling that unknown would spin Train and Video for the whole session.
test("a deferred verdict counts as settled, not as still checking", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  const selector = /capabilitiesUnknown: \(\) => \{([\s\S]*?)\n  \},/.exec(src);
  assert.ok(selector, "capabilitiesUnknown is no longer a block the deferred case can live in");
  assert.match(
    selector[1],
    /detectionDeferred/,
    "a deferred reply leaves the tabs spinning with nothing left to wait for",
  );
});

test("the sidebar gates Train and Video on a measured verdict", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /const chatOnlyMeasured = chatOnly && !capabilitiesUnknown;/,
    "the rows still read chatOnly directly, so the UA guess disables them",
  );
  // Every row-level use of the verdict goes through the measured form.
  //
  // `disabled: chatOnly` is an object entry, so the regressed form ends in a comma and
  // the old `includes("disabled: chatOnly;")` could never match anything. The character
  // class keeps `disabled: chatOnlyMeasured,` from matching, since "M" is not in it.
  for (const pattern of [/disabled: chatOnly[,;\s]/, /if \(chatOnly\) return;/]) {
    assert.ok(
      !pattern.test(src),
      `${pattern} still reads the unmeasured verdict`,
    );
  }
  // Both rows opt into the pending state rather than rendering a guessed gray-out.
  assert.equal(
    src.match(/pending: capabilitiesUnknown,/g)?.length,
    2,
    "Train and Video do not both mark themselves pending while the verdict is out",
  );
});

// beforeLoad redirects are one-way: bouncing a cold /studio or /video load to /chat on the
// pre-measurement guess strands a healthy host there for the rest of the session.
test("the route guard waits out an unknown verdict on Train and Video", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
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

// The tab is enabled on a healthy Apple Silicon host (chat_only is false there), and the
// backend has no video path for it, so the page has to say so rather than fail at load.
test("the Video page gates on the backend's own capability answer", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /hardware\.videoSupported === false/,
    "the page does not read the backend's video verdict",
  );
  assert.match(
    src,
    /capabilitiesUnknown \|\| !hardware\.loaded/,
    "the page renders its generator before the verdict has landed",
  );
  // A backend that predates the field sends nothing, which arrives as null; only an explicit
  // false may hide the generator.
  assert.ok(
    !/videoSupported !== true/.test(src),
    "an older backend's missing field would hide the page it has always served",
  );
});

// The hardcoded "needs an NVIDIA or AMD GPU" is wrong on Apple Silicon: no GPU the user can
// add would help, because there is no Apple video path at all.
test("the Video tooltip is derived, not hardcoded", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /const videoDisabledHint: string \| undefined/, "no derived video hint");
  assert.match(
    src,
    /platformDeviceType === "mac"\s*\?\s*"Video generation on macOS is coming soon\."/,
    "a Mac is still told to fit a GPU",
  );
  assert.ok(
    !/tooltip: chatOnly\s*\n?\s*\? "Video generation needs an NVIDIA or AMD GPU\."/.test(src),
    "the hardcoded tooltip is still wired to the row",
  );
});

// The Video gate waits on useHardwareInfo's `loaded`, and a failed probe used to resolve to
// DEFAULT (loaded false) with nothing scheduled to run again: one blip and the page spun for
// the rest of the session.
test("a failed hardware probe is retried, not left unloaded", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/hooks/use-hardware-info.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /if \(!cancelled && !hw\.loaded\) retry = setTimeout\(load, RETRY_MS\);/,
    "nothing re-probes after a failed read",
  );
  assert.match(
    src,
    /if \(retry !== undefined\) clearTimeout\(retry\);/,
    "the retry outlives the component that scheduled it",
  );
});

// The subscribe happens in the effect, one tick after the render read `cached`. A probe
// that resolves in that gap notifies the listeners registered at the time, which does not
// include this one, and leaves `cached` set so the fetch is skipped as redundant. Nothing
// is then scheduled to call setInfo. The PR gates whole pages on `loaded`, so the symptom
// is a permanent "Checking this machine..." rather than a stale value.
test("a cache filled between render and subscribe still reaches the component", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/hooks/use-hardware-info.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /if \(cached\) listener\(cached\);\s*\n\s*else load\(\);/,
    "a component that missed the notify has no path to the cache it skipped loading for",
  );
  // A 200 superseded by a later invalidate must not be reported as a failed probe: load()
  // reads !loaded as failure and drops the page back to its loading state.
  assert.ok(
    !/return cached \?\? DEFAULT;/.test(src),
    "a superseded but successful read still resolves as an unloaded DEFAULT",
  );
});
