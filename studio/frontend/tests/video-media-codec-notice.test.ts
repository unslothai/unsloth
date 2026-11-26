// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const notice = await import("../src/features/video/media-codec-notice.ts");
const { en } = await import("../src/i18n/locales/en.ts");

test("the command names exactly the missing packages", () => {
  assert.equal(notice.mediaCodecInstallCommand([]), null);
  assert.equal(
    notice.mediaCodecInstallCommand(["gstreamer1.0-libav", "gstreamer1.0-plugins-good"]),
    ["sudo", "apt install gstreamer1.0-libav gstreamer1.0-plugins-good"].join(" "),
  );
  assert.equal(notice.mediaCodecInstallCommand(["x; rm -rf /", "Bad"]), null);
});

test("the probe runs once per session and a failing probe shows nothing", async () => {
  notice.resetMediaCodecCheckForTests();
  const shown: string[] = [];
  let probes = 0;
  const probe = async () => {
    probes += 1;
    return ["gstreamer1.0-libav"];
  };
  await notice.checkMediaCodecsOnce(probe, (c) => shown.push(c));
  await notice.checkMediaCodecsOnce(probe, (c) => shown.push(c));
  assert.equal(probes, 1);
  assert.equal(shown.length, 1);
  assert.match(shown[0], /apt install gstreamer1\.0-libav$/);

  notice.resetMediaCodecCheckForTests();
  await notice.checkMediaCodecsOnce(
    async () => {
      throw new Error("unknown command");
    },
    (c) => shown.push(c),
  );
  notice.resetMediaCodecCheckForTests();
  await notice.checkMediaCodecsOnce(async () => [], (c) => shown.push(c));
  assert.equal(shown.length, 1);
});

test("the Video page wires the desktop probe to a toast", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /checkMediaCodecsOnce\(desktopMissingMediaCodecPackages,/);
  assert.match(page, /desktopShell\.mediaCodecs\.description/);
  assert.match(en.desktopShell.mediaCodecs.description, /\{command\}/);
  const src = readSrc("features/video/media-codec-notice.ts");
  assert.match(src, /invoke<string\[\]>\("missing_media_codec_packages"\)/);
});
