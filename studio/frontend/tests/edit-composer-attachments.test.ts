// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const thread = readSrc("components/assistant-ui/thread.tsx");
const attachment = readSrc("components/assistant-ui/attachment.tsx");

function block(source: string, start: string): string {
  const [, rest] = source.split(start, 2);
  assert.ok(rest !== undefined, `source no longer contains ${start}`);
  const [body] = (rest ?? "").split("\n};", 1);
  return body ?? "";
}

const editComposer = block(thread, "const EditComposer: FC = () => {");

test("the edit composer shows the attachments it will resend", () => {
  assert.match(
    editComposer,
    /<ComposerAttachments\b[^>]*\/>[\s\S]{0,500}<ComposerPrimitive\.Input\b/,
  );
});

test("the edit box keeps the prompt's line at full width, with the standard button pair", () => {
  const user = block(thread, "const UserMessage: FC = () => {");
  // Same top gap and type as the prompt bubble, so the text stays near its line.
  assert.match(
    user,
    /aui-user-message-root [^"]*\bpt-6 pb-4 text-ui-15p5 \[font-weight:410\]/,
  );
  assert.match(
    editComposer,
    /aui-edit-composer-wrapper [^"]*\bpt-6 pb-4 text-ui-15p5 \[font-weight:410\]/,
  );
  assert.match(
    editComposer,
    /aui-edit-composer-root flex w-full flex-col rounded-xl/,
  );
  assert.match(
    editComposer,
    /minRows=\{2\}\s*className="aui-edit-composer-input [^"]*\bpx-5 py-3\.5/,
  );
  assert.match(editComposer, /aui-edit-composer-footer mx-4 mb-4/);
  // The dialog pair: outline Cancel, dark action, both default size.
  assert.match(
    editComposer,
    /<Button type="button" variant="outline">\s*Cancel/,
  );
  assert.match(editComposer, /type="submit"\s*variant="dark"\s*disabled=/);
});

test("Send is disabled once the edit leaves nothing to send", () => {
  assert.match(
    editComposer,
    /const editEmpty = useAuiState\(\(\{ composer \}\) => composer\.isEmpty\);/,
  );
  assert.match(editComposer, /disabled=\{researchActive \|\| editEmpty\}/);
});

test("a pasted-text chip without its File previews instead of inlining", () => {
  const chip = block(attachment, "const PastedTextAttachmentUI: FC<{");
  assert.match(
    chip,
    /const canInline = isComposer && attachment\.file !== undefined;/,
  );
  assert.match(chip, /onClick=\{canInline \? showInTextField : undefined\}/);
  assert.match(
    chip,
    /\{canInline \? \(\s*chip\s*\) : \(\s*<PastedTextPreviewDialog/,
  );
  assert.match(chip, /\{isComposer && <AttachmentCardRemove \/>\}/);
});
