// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import {
  MASK_OVERLAY_RGBA,
  MASK_SEGMENT_ROUTES,
  parseMaskPrompt,
  parseSegmenterStatus,
  parseSegmentResult,
  segmenterDownloadFraction,
  segmenterDownloadNotice,
  segmenterNeedsDownload,
  segmenterPrecisionLabel,
  segmenterReady,
  segmentSummary,
  tintMaskPixels,
} from "../src/features/images/lib/mask-from-text.ts";

const MASK = "data:image/png;base64,iVBORw0KGgo=";

test("every segmenter route sits under the image inference prefix", () => {
  assert.deepEqual(MASK_SEGMENT_ROUTES, {
    status: "/api/inference/images/segment/status",
    download: "/api/inference/images/segment/download",
    cancel: "/api/inference/images/segment/cancel",
    release: "/api/inference/images/segment/release",
    segment: "/api/inference/images/segment",
  });
});

test("the status parse tolerates missing and malformed fields", () => {
  for (const raw of [null, undefined, 3, "x", [], {}]) {
    const s = parseSegmenterStatus(raw);
    assert.equal(s.supported, false);
    assert.equal(s.downloaded, false);
    assert.equal(s.size_bytes, null);
    assert.equal(s.download.downloading, false);
    assert.equal(s.download.bytes_total, null);
    assert.equal(segmenterNeedsDownload(s), false);
    assert.equal(segmenterReady(s), false);
  }
  const s = parseSegmenterStatus({
    supported: true,
    reason: null,
    downloaded: false,
    loaded: false,
    device: null,
    repo: "facebook/sam3",
    gated: true,
    license_url: "https://huggingface.co/facebook/sam3",
    size_bytes: 3_439_938_512,
    download: { downloading: true, download_id: "abc", error: null, cancelled: false, bytes_total: 100, bytes_done: 25 },
  });
  assert.equal(s.repo, "facebook/sam3");
  assert.equal(s.gated, true);
  assert.equal(s.size_bytes, 3_439_938_512);
  assert.equal(s.download.download_id, "abc");
  assert.equal(segmenterDownloadFraction(s.download), 0.25);
  assert.equal(segmenterNeedsDownload(s), true);
  // A download in flight is not ready even once the files report downloaded.
  assert.equal(segmenterReady({ ...s, downloaded: true }), false);
  assert.equal(segmenterReady({ ...s, downloaded: true, download: { ...s.download, downloading: false } }), true);
  // Negative or non-numeric sizes read as unknown.
  assert.equal(parseSegmenterStatus({ size_bytes: -1 }).size_bytes, null);
  assert.equal(segmenterDownloadFraction({ ...s.download, bytes_total: null }), null);
  assert.equal(segmenterDownloadFraction({ ...s.download, bytes_total: 10, bytes_done: 50 }), 1);
});

test("an unsupported install never offers the download", () => {
  const s = parseSegmenterStatus({ supported: false, reason: "Needs transformers 5", downloaded: false });
  assert.equal(s.reason, "Needs transformers 5");
  assert.equal(segmenterNeedsDownload(s), false);
  assert.equal(segmenterReady(s), false);
  assert.equal(segmenterNeedsDownload(null), false);
});

test("the segment result parse keeps well-formed instances and requires a mask", () => {
  const r = parseSegmentResult({
    mask: MASK,
    width: 640,
    height: 480,
    instances: [
      { prompt: "sofa", score: 0.91, box: [1, 2, 300, 200] },
      { prompt: "bad", score: "high", box: [1, 2, 3, 4] },
      { prompt: "bad box", score: 0.5, box: [1, 2] },
    ],
    prompts: [
      { text: "sofa", max_detections: 1, found: 1 },
      { text: "pillow", max_detections: 2, found: 0 },
      { text: "", max_detections: 1, found: 0 },
    ],
    device: "cuda:0",
    elapsed_ms: 412,
  });
  assert.equal(r.width, 640);
  assert.equal(r.instances.length, 1);
  assert.deepEqual(r.instances[0].box, [1, 2, 300, 200]);
  assert.equal(r.prompts.length, 2);
  assert.equal(r.device, "cuda:0");
  assert.throws(() => parseSegmentResult({}), /no mask/);
  assert.throws(() => parseSegmentResult({ mask: "not a data url" }), /no mask/);
});

test("the prompt parser matches ComfyUI's SAM 3 syntax", () => {
  assert.deepEqual(parseMaskPrompt("Sofa:2"), [["Sofa", 2]]);
  assert.deepEqual(parseMaskPrompt("(red lamp), window, pillow:3"), [
    ["red lamp", 1],
    ["window", 1],
    ["pillow", 3],
  ]);
  assert.deepEqual(parseMaskPrompt("a:2.6"), [["a", 3]]);
  assert.deepEqual(parseMaskPrompt("a:0"), [["a", 1]]);
  assert.deepEqual(parseMaskPrompt("a : 4 "), [["a", 4]]);
  assert.deepEqual(parseMaskPrompt(" , sofa,, ,"), [["sofa", 1]]);
  assert.deepEqual(parseMaskPrompt(""), []);
  // A colon not followed by a number stays part of the concept.
  assert.deepEqual(parseMaskPrompt("note: red"), [["note: red", 1]]);
  // Python's round() (half to even), as the backend parses it.
  assert.deepEqual(parseMaskPrompt("a:2.5, b:3.5, c:0.5"), [["a", 2], ["b", 4], ["c", 1]]);
  // Not a number: the whole part is the concept.
  assert.deepEqual(parseMaskPrompt("v:2.5.1"), [["v:2.5.1", 1]]);
  assert.deepEqual(parseMaskPrompt("grain:5000"), [["grain", 200]]);
});

test("the summary names what was found and what was not", () => {
  assert.equal(
    segmentSummary({
      prompts: [
        { text: "sofa", max_detections: 1, found: 1 },
        { text: "pillow", max_detections: 3, found: 2 },
        { text: "lamp", max_detections: 1, found: 0 },
      ],
      instances: [],
    }),
    "Found 2 of 3: sofa, pillow x2. Not found: lamp.",
  );
  assert.equal(
    segmentSummary({ prompts: [{ text: "cat", max_detections: 1, found: 0 }], instances: [] }),
    "Nothing found for cat.",
  );
  assert.equal(segmentSummary({ prompts: [], instances: [] }), "Nothing found.");
  for (const text of ["x"]) {
    const line = segmentSummary({ prompts: [{ text, max_detections: 1, found: 1 }], instances: [] });
    assert.ok(!/—/.test(line));
  }
});

test("the overlay tint keeps black transparent and colours white", () => {
  const layer = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255]);
  const out = tintMaskPixels(layer, MASK_OVERLAY_RGBA.mask);
  assert.deepEqual([...out.slice(0, 4)], [0, 0, 0, 0]);
  assert.deepEqual([...out.slice(4, 8)], [244, 114, 114, 140]);
  assert.equal(out[11], 70);
  assert.deepEqual([...tintMaskPixels(layer, MASK_OVERLAY_RGBA.paint).slice(4, 8)], [255, 255, 255, 255]);
});

test("both painters take a text mask and the page renders the panel for inpaint and mask or paint edits", () => {
  const page = readSrc("features/images/images-page.tsx");
  const localized = readSrc("features/images/localized-edit-canvas.tsx");
  const apply = readSrc("features/images/mask-apply.ts");
  assert.match(page, /function MaskCanvas\([\s\S]*?applyMask\?: AppliedMask \| null;/);
  assert.match(localized, /applyMask\?: AppliedMask \| null;/);
  // Not for annotate: its layer is coloured outlines, not a mask.
  assert.match(localized, /mode === "annotate"\) return;/);
  assert.match(page, /onMaskChange=\{setMaskImage\}\s*applyMask=\{textMask\}\s*\/>\s*<\/Field>\s*<MaskFromTextPanel image=\{initImage\}/);
  assert.match(page, /applyMask=\{textMask\}\s*\/>\s*\{localizedMode !== "annotate" && \(\s*<MaskFromTextPanel/);
  // Replace clears to black first; additive unions with "lighten".
  assert.match(apply, /if \(!additive\) \{[\s\S]*?fillRect/);
  assert.match(apply, /globalCompositeOperation = "lighten"/);
  // Each painter replays only results that arrive after it mounted.
  assert.equal((page.match(/useRef<number \| null>\(applyMask\?\.key \?\? null\)/g) ?? []).length, 1);
  assert.equal((localized.match(/useRef<number \| null>\(applyMask\?\.key \?\? null\)/g) ?? []).length, 1);
});

test("the panel releases the segmenter on unmount and uses the shared download idioms", () => {
  const panel = readSrc("features/images/mask-from-text-panel.tsx");
  const api = readSrc("features/images/mask-segment-api.ts");
  assert.match(panel, /releaseSegmenter\(\)\.catch/);
  assert.match(panel, /useIsAccountOwner/);
  assert.match(panel, /hfApiToken\(getHfToken\(\)\)/);
  assert.match(api, /MASK_SEGMENT_ROUTES\.segment/);
  assert.match(api, /readFastApiError/);
  for (const src of [panel, api]) assert.ok(!/—/.test(src), "no em dashes in UI copy");
});

test("a text mask that went stale while it decoded never touches the reused canvases", async () => {
  const { applyMaskToLayer } = await import("../src/features/images/mask-apply.ts");
  const g = globalThis as Record<string, unknown>;
  const saved = { Image: g.Image, ImageData: g.ImageData };
  // The decode resolves on a later tick, as a real <img> does.
  g.Image = class {
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    set src(_v: string) {
      setTimeout(() => this.onload?.(), 0);
    }
  };
  g.ImageData = class {
    data: Uint8ClampedArray;
    constructor(data: Uint8ClampedArray) {
      this.data = data;
    }
  };
  const calls: string[] = [];
  const ctx = new Proxy(
    {},
    {
      get: (_t, key) =>
        key === "getImageData"
          ? () => ({ data: new Uint8ClampedArray(4) })
          : typeof key === "string" && /^[a-z]+[A-Z]|^(save|restore|fillRect|drawImage)$/.test(key)
            ? () => calls.push(key)
            : undefined,
      set: (_t, key) => {
        calls.push(`set:${String(key)}`);
        return true;
      },
    },
  );
  const canvas = () =>
    ({ width: 1, height: 1, getContext: () => ctx, toDataURL: () => "data:image/png;base64,AA==" }) as unknown as HTMLCanvasElement;
  try {
    let live = true;
    const stale = applyMaskToLayer({
      layer: canvas(),
      display: canvas(),
      dataUrl: MASK,
      additive: false,
      overlay: "mask",
      isCurrent: () => live,
    });
    live = false; // Clear, a new source or a newer mask before the decode finished
    assert.equal(await stale, null);
    // Not deepEqual(calls, []): strict assert narrows calls to never[] after it, so the later includes() checks
    // stop typechecking.
    assert.equal(calls.length, 0);
    const fresh = await applyMaskToLayer({
      layer: canvas(),
      display: canvas(),
      dataUrl: MASK,
      additive: false,
      overlay: "mask",
      isCurrent: () => true,
    });
    assert.equal(fresh, "data:image/png;base64,AA==");
    assert.ok(calls.includes("drawImage") && calls.includes("putImageData"));
  } finally {
    g.Image = saved.Image;
    g.ImageData = saved.ImageData;
  }
  // Both painters pass their effect's liveness.
  assert.match(readSrc("features/images/images-page.tsx"), /overlay: "mask",\s*isCurrent: \(\) => live,/);
  assert.match(readSrc("features/images/localized-edit-canvas.tsx"), /isCurrent: \(\) => live,/);
});

test("a Replace that found nothing leaves both painters holding no mask", () => {
  const panel = readSrc("features/images/mask-from-text-panel.tsx");
  const page = readSrc("features/images/images-page.tsx");
  const localized = readSrc("features/images/localized-edit-canvas.tsx");
  // The canvas still clears (the black PNG is applied), but the result is flagged as selecting nothing...
  assert.match(panel, /onMask\(\{ dataUrl: result\.mask, additive, empty: result\.instances\.length === 0 \}\)/);
  // ...so the logical value stays null and the page's !maskImage / !localizedLayer checks still refuse Generate.
  assert.match(page, /onMaskChange\(applyMask\.empty \? null : url\)/);
  assert.match(localized, /dirty\.current = !applyMask\.empty;\s*onLayerChange\(applyMask\.empty \? null : url\)/);
});

test("the status parse reads the active variant, and an older backend reads as SAM 3", () => {
  const sam31 = parseSegmenterStatus({
    supported: true,
    downloaded: false,
    variant: "sam3.1",
    label: "SAM 3.1",
    repo: "unsloth/sam3.1",
    revision: "main",
    gated: false,
    license_url: "https://huggingface.co/unsloth/sam3.1/blob/main/LICENSE",
    size_bytes: 1_685_972_909,
    download: {},
  });
  assert.equal(sam31.variant, "sam3.1");
  assert.equal(sam31.label, "SAM 3.1");
  assert.equal(sam31.revision, "main");
  assert.equal(sam31.gated, false);
  const old = parseSegmenterStatus({ supported: true, repo: "facebook/sam3", gated: true });
  assert.equal(old.variant, "");
  assert.equal(old.label, "SAM 3");
  assert.equal(old.revision, "");
});

test("the download notice names the variant and asks for a license only when it is gated", () => {
  const ungated = segmenterDownloadNotice({ label: "SAM 3.1", gated: false }, "1.7 GB");
  assert.equal(ungated.text, "Mask from text uses SAM 3.1 (1.7 GB), which is not downloaded yet.");
  assert.equal(ungated.needsLicense, false);
  const gated = segmenterDownloadNotice({ label: "SAM 3", gated: true }, "3.4 GB");
  assert.equal(gated.text, "Mask from text uses SAM 3 (3.4 GB), which is not downloaded yet.");
  assert.equal(gated.needsLicense, true);
  assert.equal(
    segmenterDownloadNotice({ label: "", gated: false }, null).text,
    "Mask from text uses SAM 3, which is not downloaded yet.",
  );
});

test("the download notice names the precision it would fetch", () => {
  assert.equal(
    segmenterDownloadNotice({ label: "SAM 3.1", gated: false, precision: "int8" }, "0.9 GB").text,
    "Mask from text uses SAM 3.1 (INT8, 0.9 GB), which is not downloaded yet.",
  );
  assert.equal(
    segmenterDownloadNotice({ label: "SAM 3.1", gated: false, precision: "bf16" }, null).text,
    "Mask from text uses SAM 3.1 (BF16), which is not downloaded yet.",
  );
  // An unknown or missing precision (an older backend) keeps the plain notice.
  assert.equal(
    segmenterDownloadNotice({ label: "SAM 3.1", gated: false, precision: "nf4" }, "1.7 GB").text,
    "Mask from text uses SAM 3.1 (1.7 GB), which is not downloaded yet.",
  );
  assert.equal(segmenterPrecisionLabel("fp8"), "FP8");
  assert.equal(segmenterPrecisionLabel(null), null);
  assert.equal(parseSegmenterStatus({ precision: "int8" }).precision, "int8");
  assert.equal(parseSegmenterStatus({}).precision, null);
});

test("the panel takes the notice text and the license hint from the active variant", () => {
  const src = readSrc("features/images/mask-from-text-panel.tsx");
  assert.match(src, /segmenterDownloadNotice\(status, size\)/);
  assert.match(src, /notice\.needsLicense \?/);
  assert.doesNotMatch(src, /status\.gated \?/);
  assert.doesNotMatch(src, /"Could not download SAM 3"|Downloading SAM 3/);
});
