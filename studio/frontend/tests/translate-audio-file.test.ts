// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS } from "../src/features/chat/open-document-accept.ts";
import {
  AUDIO_ACCEPT_EXTENSIONS,
  isAudioAttachmentFile,
} from "../src/lib/audio-utils.ts";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/translate/translate-page.tsx");
const audio = readSrc("features/translate/translate-audio.ts");

function quoted(name: string): string {
  const match = new RegExp(`const ${name} =\\s*"([^"]+)"`).exec(page);
  assert.ok(match, name);
  return match[1];
}

test("documents, images and text files keep their own reader", () => {
  const images = /"(\.png[^"]*)"/.exec(page)?.[1];
  assert.ok(images);
  const exts = [
    quoted("TEXT_FILE_EXTENSIONS"),
    ".pdf,.docx,.xlsx,.pptx,.odt,.ods,.rtf,.eml",
    SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS,
    images,
  ]
    .join(",")
    .split(",");
  for (const ext of exts) {
    const type = ext === ".txt" ? "text/plain" : "";
    assert.equal(
      isAudioAttachmentFile({ name: `file${ext}`, type }),
      false,
      ext,
    );
  }
});

test("every audio file the drop zone takes is transcribed, not sent to the document reader", () => {
  for (const ext of AUDIO_ACCEPT_EXTENSIONS.split(",")) {
    assert.equal(
      isAudioAttachmentFile({ name: `Clip${ext.toUpperCase()}`, type: "" }),
      true,
      ext,
    );
  }
  assert.equal(
    isAudioAttachmentFile({ name: "voice", type: "audio/mp4" }),
    true,
  );
  assert.equal(
    isAudioAttachmentFile({ name: "clip.mp4", type: "video/mp4" }),
    false,
  );
  assert.match(
    page,
    /const DROP_ACCEPT = `\$\{OPEN_FILE_EXTENSIONS\},\$\{AUDIO_ACCEPT_EXTENSIONS\},\$\{VIDEO_FILE_EXTENSIONS\}`;/,
  );
  assert.match(page, /accept: DROP_ACCEPT,/);
  assert.match(page, /accept=\{PICKER_ACCEPT\}/);
  assert.match(
    page,
    /if \(!file\) return;\s*if \(isTranscribedFile\(file\)\) \{\s*await openAudio\(file\);\s*return;\s*\}\s*if \(isTextFile\(file\)\)/,
  );
});

test("an audio file is transcribed with the Voice settings model and fills the text box", () => {
  assert.match(
    audio,
    /transcribeAudioBlob\(file, \{ model, ggufVariant, signal \}\)/,
  );
  assert.match(
    audio,
    /const ggufVariant = sttModelVariant\(model, settings\.sttGgufVariant\);/,
  );
  assert.match(audio, /requestSttDownload\(model, \{ ggufVariant \}\);/);
  // A text typed or picked from history while the file transcribes wins.
  assert.match(
    page,
    /if \(text && useTranslateStore\.getState\(\)\.source === before\) \{\s*loadText\(text\);/,
  );
  assert.match(
    page,
    /useEffect\(\(\) => \(\) => audioAbort\.current\?\.abort\(\), \[\]\);/,
  );
});
