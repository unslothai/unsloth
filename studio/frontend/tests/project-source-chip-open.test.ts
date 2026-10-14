// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";

import type * as ChipModule from "../src/features/rag/components/document-status-chip.tsx";
import type * as SourceChipModule from "../src/features/rag/components/project-source-chip.tsx";
import { type StubElement, loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

const Nothing = () => null;

const chip = loadWithStubs<typeof ChipModule>(
  new URL("../src/features/rag/components/document-status-chip.tsx", import.meta.url),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/assistant-ui/badge": {
      Badge: ({ children }: { children?: React.ReactNode }) =>
        React.createElement("span", { "data-badge": "" }, children),
    },
    "@/components/ui/spinner": { Spinner: Nothing },
    "@/lib/utils": { cn: (...parts: unknown[]) => parts.filter(Boolean).join(" ") },
    "@hugeicons/core-free-icons": {},
    "@hugeicons/react": { HugeiconsIcon: Nothing },
    "lucide-react": { XIcon: Nothing },
  },
);

const LocalFileDialog = () => null;

const { ProjectSourceChip } = loadWithStubs<typeof SourceChipModule>(
  new URL("../src/features/rag/components/project-source-chip.tsx", import.meta.url),
  {
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/assistant-ui/attachment-document-dialog": { LocalFileDialog },
    "@/components/file-viewer": { MAX_DOCUMENT_PREVIEW_BYTES: 1 },
    "@/features/chat": {
      attachmentFileKind: (name: string) => (name.endsWith(".pdf") ? "pdf" : "file"),
    },
    "./document-status-chip": chip,
    "./source-file": {
      asViewerFile: (file: File) => file,
      fetchSourceFile: async () => null,
      isViewableSource: (name: string) => /\.(pdf|md|txt|docx)$/.test(name),
    },
  },
);

function chipProps(doc: Record<string, unknown>): Record<string, unknown> {
  const element = ProjectSourceChip({
    doc: doc as unknown as Parameters<typeof ProjectSourceChip>[0]["doc"],
  }) as unknown as StubElement;
  assert.equal(element.type, chip.DocumentStatusChip);
  return element.props;
}

// PDFs used to open a separate preview sheet; every source now opens in the attachment viewer.
test("an indexed PDF source opens in the attachment viewer", () => {
  const props = chipProps({ id: "doc-1", filename: "handbook.pdf", status: "completed" });
  assert.equal(props.onOpen, undefined);
  const trigger = React.createElement("button");
  const wrapped = (props.openTrigger as (t: React.ReactElement) => StubElement)(trigger);
  assert.equal(wrapped.type, LocalFileDialog);
  assert.equal(wrapped.props.name, "handbook.pdf");
});

test("an indexed text, markdown or Office source opens in the attachment viewer", () => {
  for (const filename of ["notes.md", "notes.txt", "Docs/report.docx"]) {
    const props = chipProps({ id: "doc-2", filename, status: "completed" });
    assert.equal(props.onOpen, undefined);
    const trigger = React.createElement("button");
    const wrapped = (props.openTrigger as (t: React.ReactElement) => StubElement)(trigger);
    assert.equal(wrapped.type, LocalFileDialog);
    assert.equal(wrapped.props.name, filename.split("/").pop());
    assert.equal(wrapped.props.children, trigger);
  }
});

test("sources still indexing, failed, optimistic or unviewable stay inert", () => {
  for (const doc of [
    { id: "doc-3", filename: "handbook.pdf", status: "running" },
    { id: "doc-3", filename: "notes.md", status: "pending" },
    { id: "doc-3", filename: "notes.md", status: "failed" },
    { id: "pending_1", filename: "notes.md", status: "completed" },
    { id: "doc-3", filename: "archive.epub", status: "completed" },
  ]) {
    const props = chipProps(doc);
    assert.equal(props.onOpen, undefined, JSON.stringify(doc));
    assert.equal(props.openTrigger, undefined, JSON.stringify(doc));
  }
});

test("remove and retry pass through to the chip unchanged", () => {
  const onRemove = () => undefined;
  const onRetry = () => undefined;
  const element = ProjectSourceChip({
    doc: { id: "doc-4", filename: "notes.md", status: "failed" } as unknown as Parameters<
      typeof ProjectSourceChip
    >[0]["doc"],
    onRemove,
    onRetry,
  }) as unknown as StubElement;
  assert.equal(element.props.onRemove, onRemove);
  assert.equal(element.props.onRetry, onRetry);
});

test("the chip name is a button only when the source can be opened", () => {
  const render = (props: Record<string, unknown>) =>
    renderToStaticMarkup(
      React.createElement(chip.DocumentStatusChip, {
        filename: "notes.md",
        status: "completed",
        ...props,
      } as Parameters<typeof chip.DocumentStatusChip>[0]),
    );
  assert.doesNotMatch(render({}), /<button/);
  assert.match(render({ onOpen: () => undefined }), /<button type="button" aria-label="Preview notes.md"/);
  assert.match(
    render({
      openTrigger: (t: React.ReactElement) => React.createElement("i", { "data-viewer": "" }, t),
    }),
    /<i data-viewer=""><button type="button" aria-label="Preview notes.md"/,
  );
  // A still-indexing row keeps its spinner and no open affordance, even when asked.
  assert.doesNotMatch(render({ status: "running", onOpen: () => undefined }), /Preview notes/);
});

test("the project Sources tab renders openable chips", () => {
  const panel = readFileSync(
    new URL("../src/features/rag/components/project-sources-panel.tsx", import.meta.url),
    "utf8",
  );
  // Each file is openable; a linked folder stays one summary chip.
  assert.match(panel, /groups\.loose\.map\(\(doc\) => \(\s*<ProjectSourceChip\s+key=\{doc\.id\}\s+doc=\{doc\}/);
});
