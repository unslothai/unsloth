// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { unzipSync } from "fflate";
import {
  MAX_OPEN_DOCUMENT_ARCHIVE_BYTES,
  MAX_OPEN_DOCUMENT_XML_BYTES,
} from "./open-document";

// iWork 2013 and later keep a document as protobuf records in Snappy-compressed Index/*.iwa files.
// No schema ships with them; the message types and field numbers below are the ones iWork writes.
// Calls, not consts: read at module scope, an import the chat barrel also reaches can still be in
// its temporal dead zone when this module runs.
function maxUnpackedBytes(): number {
  return 2 * MAX_OPEN_DOCUMENT_ARCHIVE_BYTES;
}
function maxTextLength(): number {
  return MAX_OPEN_DOCUMENT_XML_BYTES;
}
// Charged per stored object, so a flood of empty records cannot outgrow the unpacked budget.
const OBJECT_OVERHEAD_BYTES = 256;
const MAX_REFERENCE_DEPTH = 8;

const DOCUMENT = 1;
const SHOW = 2;
const SLIDE_NODE = 4;
const SLIDE = 5;
const STYLESHEET = 401;
const TEXT_STORAGE = 2001;
const PAGES_DOCUMENT = 10000;
// Objects that link back to the deck, a slide or the styles: following them would repeat another
// slide's text or walk the whole stylesheet.
const WALK_STOPS = new Set([DOCUMENT, SHOW, SLIDE_NODE, SLIDE, STYLESHEET]);
// Title and body placeholders, then drawables. The rest are builds, whose order is not reading
// order, and the template slide and speaker notes, which the PPTX reader also leaves out.
const SLIDE_TEXT_FIELDS = [5, 6, 7];

export type IworkAttachmentContent = {
  label: "PAGES" | "KEY";
  text: string;
};

type IworkObject = { type: number; payload: Uint8Array };
type Objects = Map<number, IworkObject>;
type Field = {
  field: number;
  wire: number;
  value: number;
  bytes: Uint8Array | null;
};
type Budget = { remaining: number };
// Every object is walked at most once per document, so structure that several slides share, or a
// slide tree that lists a node twice, cannot multiply the work.
type Walk = { objects: Objects; seen: Set<number> };

class IworkSizeError extends Error {}

const utf8 = new TextDecoder();

export async function readIworkAttachmentContent(
  file: File,
  filename: string,
): Promise<IworkAttachmentContent> {
  const objects = await readObjects(file);
  const types = new Set([...objects.values()].map((object) => object.type));
  if (types.has(PAGES_DOCUMENT)) {
    return { label: "PAGES", text: pagesText(objects, filename) };
  }
  return { label: "KEY", text: keynoteText(objects, filename) };
}

async function readObjects(file: File): Promise<Objects> {
  if (file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES) {
    throw new IworkSizeError(`iWork file is too large: ${file.name}`);
  }
  const budget = { remaining: maxUnpackedBytes() };
  const objects: Objects = new Map();
  try {
    const entries = unzipSync(new Uint8Array(await file.arrayBuffer()), {
      filter: (entry) => {
        if (!/^Index\/.+\.iwa$/.test(entry.name)) {
          return false;
        }
        // Charged like the Office reader: fflate may allocate either size.
        charge(budget, Math.max(entry.size, entry.originalSize), file.name);
        return true;
      },
    });
    for (const data of Object.values(entries)) {
      readArchives(
        iwaData(data, budget, file.name),
        objects,
        budget,
        file.name,
      );
    }
  } catch (error) {
    if (error instanceof IworkSizeError) {
      throw error;
    }
    throw new Error(`Failed to read iWork file: ${file.name}`, {
      cause: error,
    });
  }
  if (objects.size === 0) {
    throw new Error(
      `Not an iWork document saved by iWork 2013 or later: ${file.name}`,
    );
  }
  return objects;
}

function charge(budget: Budget, bytes: number, name: string): void {
  budget.remaining -= bytes;
  if (budget.remaining < 0) {
    throw new IworkSizeError(`iWork file unpacks too large: ${name}`);
  }
}

function* iwaChunks(data: Uint8Array): Generator<Uint8Array> {
  for (let offset = 0; offset < data.length;) {
    if (data[offset] !== 0 || offset + 4 > data.length) {
      throw new Error("Unexpected IWA chunk header");
    }
    const end = offset + 4 + littleEndian(data, offset + 1, 3);
    if (end > data.length) {
      throw new Error("Truncated IWA chunk");
    }
    yield data.subarray(offset + 4, end);
    offset = end;
  }
}

/** Every chunk inflated into one buffer: a run of empty chunks costs no allocation each. */
function iwaData(data: Uint8Array, budget: Budget, name: string): Uint8Array {
  let total = 0;
  for (const chunk of iwaChunks(data)) {
    total += varint(chunk, 0)[0];
  }
  charge(budget, total, name);
  const output = new Uint8Array(total);
  let written = 0;
  for (const chunk of iwaChunks(data)) {
    written = snappy(chunk, output, written);
  }
  return output;
}

/** Inflates one Snappy block into `output` at `start`, returning where it ends. A copy stays
 *  inside `output`, though a corrupt one may reach into an earlier block. */
function snappy(input: Uint8Array, output: Uint8Array, start: number): number {
  const [length, first] = varint(input, 0);
  const end = start + length;
  let written = start;
  for (let offset = first; offset < input.length;) {
    const tag = input[offset++]!;
    if ((tag & 3) === 0) {
      let size = tag >> 2;
      if (size >= 60) {
        const bytes = size - 59;
        size = littleEndian(input, offset, bytes);
        offset += bytes;
      }
      size += 1;
      if (offset + size > input.length || written + size > end) {
        throw new Error("Corrupt Snappy literal");
      }
      output.set(input.subarray(offset, offset + size), written);
      offset += size;
      written += size;
      continue;
    }
    let size: number;
    let distance: number;
    if ((tag & 3) === 1) {
      size = ((tag >> 2) & 7) + 4;
      distance = ((tag >> 5) << 8) | littleEndian(input, offset, 1);
      offset += 1;
    } else {
      const bytes = (tag & 3) === 2 ? 2 : 4;
      size = (tag >> 2) + 1;
      distance = littleEndian(input, offset, bytes);
      offset += bytes;
    }
    if (distance === 0 || distance > written || written + size > end) {
      throw new Error("Corrupt Snappy copy");
    }
    // Byte by byte: a copy may overlap the bytes it is still writing.
    for (let index = 0; index < size; index++, written++) {
      output[written] = output[written - distance]!;
    }
  }
  if (written !== end) {
    throw new Error("Corrupt Snappy block");
  }
  return end;
}

function littleEndian(data: Uint8Array, offset: number, bytes: number): number {
  if (offset + bytes > data.length) {
    throw new Error("Truncated little-endian value");
  }
  let value = 0;
  for (let index = 0; index < bytes; index++) {
    value += data[offset + index]! * 2 ** (8 * index);
  }
  return value;
}

function varint(data: Uint8Array, offset: number): [number, number] {
  let value = 0;
  for (let shift = 0; shift < 70; shift += 7) {
    const byte = data[offset++];
    if (byte === undefined) {
      throw new Error("Truncated varint");
    }
    value += (byte & 0x7f) * 2 ** shift;
    if (byte < 0x80) {
      return [value, offset];
    }
  }
  throw new Error("Varint too long");
}

/** Parsed lazily and never kept: a message of millions of tiny fields costs no memory per field. */
function* fields(message: Uint8Array): Generator<Field> {
  for (let offset = 0; offset < message.length;) {
    const [key, next] = varint(message, offset);
    const field = Math.floor(key / 8);
    const wire = key % 8;
    offset = next;
    if (field === 0) {
      throw new Error("Invalid protobuf field");
    }
    if (wire === 0) {
      const [value, after] = varint(message, offset);
      offset = after;
      yield { field, wire, value, bytes: null };
      continue;
    }
    let size: number;
    if (wire === 2) {
      [size, offset] = varint(message, offset);
    } else if (wire === 1 || wire === 5) {
      size = wire === 1 ? 8 : 4;
    } else {
      throw new Error(`Unsupported protobuf wire type ${wire}`);
    }
    if (offset + size > message.length) {
      throw new Error("Truncated protobuf field");
    }
    offset += size;
    yield {
      field,
      wire,
      value: 0,
      bytes: message.subarray(offset - size, offset),
    };
  }
}

function numberField(message: Uint8Array, field: number): number {
  for (const f of fields(message)) {
    if (f.field === field && f.wire === 0) {
      return f.value;
    }
  }
  return 0;
}

function bytesField(message: Uint8Array, field: number): Uint8Array | null {
  for (const f of fields(message)) {
    if (f.field === field && f.wire === 2) {
      return f.bytes;
    }
  }
  return null;
}

/** An ArchiveInfo, then its payloads. */
function readArchives(
  data: Uint8Array,
  objects: Objects,
  budget: Budget,
  name: string,
): void {
  for (let offset = 0; offset < data.length;) {
    const [length, start] = varint(data, offset);
    if (start + length > data.length) {
      throw new Error("Truncated IWA record");
    }
    const info = data.subarray(start, start + length);
    offset = start + length;
    const identifier = numberField(info, 1);
    let first = true;
    for (const message of fields(info)) {
      if (message.field !== 2 || !message.bytes) {
        continue;
      }
      const size = numberField(message.bytes, 3);
      if (offset + size > data.length) {
        throw new Error("Truncated IWA payload");
      }
      if (first && !objects.has(identifier)) {
        charge(budget, OBJECT_OVERHEAD_BYTES, name);
        objects.set(identifier, {
          type: numberField(message.bytes, 1),
          payload: data.subarray(offset, offset + size),
        });
      }
      first = false;
      offset += size;
    }
  }
}

function referenceId(bytes: Uint8Array, objects: Objects): number | null {
  let id: number | null = null;
  try {
    for (const f of fields(bytes)) {
      if (id !== null || f.field !== 1 || f.wire !== 0) {
        return null;
      }
      id = f.value;
    }
  } catch {
    return null;
  }
  return id !== null && objects.has(id) ? id : null;
}

function isMessage(bytes: Uint8Array): boolean {
  try {
    const parsed = fields(bytes);
    while (!parsed.next().done);
    return true;
  } catch {
    return false;
  }
}

/** Objects a message points at, in field order, so a caller wanting the first reads no further. */
function* references(
  message: Uint8Array,
  objects: Objects,
  field?: number,
  depth = 0,
): Generator<number> {
  if (depth === 0 && !isMessage(message)) {
    return;
  }
  for (const f of fields(message)) {
    if (f.wire !== 2 || (field !== undefined && f.field !== field)) {
      continue;
    }
    const id = referenceId(f.bytes!, objects);
    if (id !== null) {
      yield id;
    } else if (depth < MAX_REFERENCE_DEPTH && isMessage(f.bytes!)) {
      yield* references(f.bytes!, objects, undefined, depth + 1);
    }
  }
}

function findObject(objects: Objects, type: number): IworkObject | undefined {
  for (const object of objects.values()) {
    if (object.type === type) {
      return object;
    }
  }
  return undefined;
}

function storageText(payload: Uint8Array): string {
  // Joined as bytes and decoded once, so a storage of many tiny runs costs no string per run.
  const text = new Uint8Array(payload.length);
  let length = 0;
  for (const f of fields(payload)) {
    if (f.field === 3 && f.wire === 2) {
      text.set(f.bytes!, length);
      length += f.bytes!.length;
    }
  }
  return (
    utf8
      .decode(text.subarray(0, length))
      // U+FFFC holds the place of an inline object: a slide number, an image, a footnote mark.
      .replace(/\uFFFC/g, "")
      .replace(/[\u2028\u2029]/g, "\n")
      .replace(/\p{Cc}/gu, (control) =>
        control === "\t" || control === "\n" ? control : "",
      )
      .trim()
  );
}

function pushLine(lines: string[], line: string, budget: Budget): void {
  if (line.trim() && budget.remaining > 0) {
    lines.push(line.slice(0, budget.remaining));
    budget.remaining -= line.length + 1;
  }
}

/** Text boxes reachable from one object, in the order its fields name them. */
function collect(
  id: number,
  walk: Walk,
  lines: string[],
  budget: Budget,
): void {
  const object = walk.objects.get(id);
  if (
    !object ||
    WALK_STOPS.has(object.type) ||
    walk.seen.has(id) ||
    budget.remaining <= 0
  ) {
    return;
  }
  walk.seen.add(id);
  if (object.type === TEXT_STORAGE) {
    pushLine(lines, storageText(object.payload), budget);
  } else {
    for (const next of references(object.payload, walk.objects)) {
      collect(next, walk, lines, budget);
    }
  }
}

function truncated(sections: string[], budget: Budget, kind: string): string {
  if (budget.remaining <= 0) {
    sections.push(
      `[Truncated: the ${kind} has more text than one attachment carries]`,
    );
  }
  return sections.join("\n\n");
}

function pagesText(objects: Objects, filename: string): string {
  const document = findObject(objects, PAGES_DOCUMENT);
  const [bodyId] = document ? references(document.payload, objects, 4) : [];
  const body = objects.get(bodyId ?? -1);
  if (body?.type !== TEXT_STORAGE) {
    throw new Error(`Pages document has no body text: ${filename}`);
  }
  const budget = { remaining: maxTextLength() };
  const lines: string[] = [];
  pushLine(lines, storageText(body.payload), budget);
  return truncated(lines, budget, "document");
}

function keynoteText(objects: Objects, filename: string): string {
  const show = findObject(objects, SHOW);
  const tree = show && bytesField(show.payload, 3);
  if (!tree) {
    throw new Error(`Keynote file has no slide list: ${filename}`);
  }
  const budget = { remaining: maxTextLength() };
  const walk = { objects, seen: new Set<number>() };
  const slides: string[] = [];
  let number = 0;
  // Skipped slides keep their number, so [Slide N] matches Keynote's own numbering.
  const visit = (nodeId: number): void => {
    const node = objects.get(nodeId);
    if (
      node?.type !== SLIDE_NODE ||
      walk.seen.has(nodeId) ||
      budget.remaining <= 0
    ) {
      return;
    }
    walk.seen.add(nodeId);
    number += 1;
    const [slideId = -1] = references(node.payload, objects, 2);
    const slide = objects.get(slideId);
    if (
      slide &&
      !walk.seen.has(slideId) &&
      numberField(node.payload, 4) !== 1
    ) {
      walk.seen.add(slideId);
      const lines: string[] = [];
      for (const field of SLIDE_TEXT_FIELDS) {
        for (const id of references(slide.payload, objects, field)) {
          collect(id, walk, lines, budget);
        }
      }
      if (lines.length > 0) {
        slides.push(`[Slide ${number}]\n${lines.join("\n")}`);
      }
    }
    for (const child of references(node.payload, objects, 1)) {
      visit(child);
    }
  };
  for (const nodeId of references(tree, objects, 2)) {
    visit(nodeId);
  }
  return truncated(slides, budget, "presentation");
}
