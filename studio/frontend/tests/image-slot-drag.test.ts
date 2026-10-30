// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyNativeImageDrop,
  draggedImageSlot,
  endImageSlotDrag,
  startImageSlotDrag,
} from "../src/components/image-slot-drag.ts";
import { readSrc } from "./helpers/kit.ts";

const IMAGE_DROPZONE = readSrc("components/image-dropzone.tsx");
const SLOT_DRAG = "../src/components/image-slot-drag.ts";

test("an image dragged out of the page is not reported as a bad file format", () => {
  for (const path of [
    "data:image/png;base64,iVBORw0KGgo=",
    "blob:http://127.0.0.1:8888/1f2e",
    "http://127.0.0.1:8888/api/images/out.png",
  ]) {
    assert.equal(classifyNativeImageDrop(path), "in-app", path);
  }
});

test("real files are classified by extension", () => {
  assert.equal(classifyNativeImageDrop("/home/me/cat.PNG"), "image");
  assert.equal(classifyNativeImageDrop("C:\\Users\\me\\cat.webp"), "image");
  assert.equal(classifyNativeImageDrop("/home/me/cat.heic"), "unsupported");
  assert.equal(classifyNativeImageDrop("/home/me/notes"), "unsupported");
  assert.equal(classifyNativeImageDrop(undefined), "none");
});

test("a slot image outlives dragend just long enough for the native drop", () => {
  startImageSlotDrag("data:image/png;base64,AAAA");
  assert.equal(draggedImageSlot(), "data:image/png;base64,AAAA");
  endImageSlotDrag(1_000);
  assert.equal(draggedImageSlot(1_500), "data:image/png;base64,AAAA");
  assert.equal(draggedImageSlot(2_500), null);
  assert.equal(draggedImageSlot(1_500), null);
});

test("a new drag replaces the previous slot image", () => {
  startImageSlotDrag("data:image/png;base64,AAAA");
  endImageSlotDrag(1_000);
  startImageSlotDrag("data:image/png;base64,BBBB");
  assert.equal(draggedImageSlot(9_000), "data:image/png;base64,BBBB");
  endImageSlotDrag(0);
  draggedImageSlot(10_000);
});

test("a filled slot drags its image and an empty slot copies it in", () => {
  assert.match(
    IMAGE_DROPZONE,
    /draggable=\{true\}[\s\S]*?startImageSlotDrag\(value\)/,
  );
  assert.match(IMAGE_DROPZONE, /onDragEnd=\{\(\) => endImageSlotDrag\(\)\}/);
  assert.match(IMAGE_DROPZONE, /<img\s+src=\{value\}\s+draggable=\{false\}/);
  assert.match(
    IMAGE_DROPZONE,
    /onDrop=\{\(e\) => \{[\s\S]*?isSlotDrag\(e\) \? draggedImageSlot\(\) : null;[\s\S]*?onChange\(slotImage\)/,
  );
});

test("a native drop with no path or an in-app URL takes the dragged slot image", () => {
  assert.match(
    IMAGE_DROPZONE,
    /const kind = classifyNativeImageDrop\(path\);\s*const slotImage =\s*kind === "none" \|\| kind === "in-app" \? draggedImageSlot\(\) : null;[\s\S]*?onChange\(slotImage\);\s*return;\s*\}\s*if \(!path\) return;/,
  );
});

test("an in-app native drop never shows the file format error", () => {
  assert.doesNotMatch(
    IMAGE_DROPZONE,
    /Other image formats can still be chosen with the picker/,
  );
  assert.match(
    IMAGE_DROPZONE,
    /if \(kind === "in-app"\) \{\s*toast\.error\("Can't drop this image here"/,
  );
});

// macOS drops arrive pathless, so a stale slot image must not claim the next drag.
test("any other drag drops the slot image", async () => {
  const doc = new EventTarget();
  Object.assign(globalThis, { document: doc });
  const slot = (await import(
    `${SLOT_DRAG}?other-drag`
  )) as typeof import("../src/components/image-slot-drag.ts");
  slot.startImageSlotDrag("data:image/png;base64,AAAA");
  slot.endImageSlotDrag(1_000);
  doc.dispatchEvent(new Event("dragstart"));
  assert.equal(slot.draggedImageSlot(1_200), null);
  doc.dispatchEvent(new Event("dragstart"));
  slot.startImageSlotDrag("data:image/png;base64,BBBB");
  assert.equal(slot.draggedImageSlot(), "data:image/png;base64,BBBB");
  Reflect.deleteProperty(globalThis, "document");
});
