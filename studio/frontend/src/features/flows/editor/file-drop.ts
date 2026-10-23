// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { uploadAsset } from "../api";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import {
  MAX_TEXT_BYTES,
  inputForAssetKind,
  inputForFile,
} from "./canvas-rules";
import { createNode } from "./node-factory";

/** Turn files into input blocks at `at` (flow coordinates), stacked downwards. Returns the new node ids. */
export async function addFileNodes(
  files: File[],
  at: { x: number; y: number },
): Promise<string[]> {
  const added: string[] = [];
  let skipped = 0;
  // An upload can finish after the user opened another flow; its block belongs to the flow it was dropped on.
  const flowId = useDocStore.getState().flowId;
  for (const [i, file] of files.entries()) {
    const target = inputForFile(file.name, file.type);
    if (!target) {
      skipped += 1;
      continue;
    }
    const position: [number, number] = [
      Math.round(at.x - 130),
      Math.round(at.y - 20 + i * 140),
    ];
    let node: FlowNode | null = createNode(target.type, position);
    if (!node) continue;
    try {
      if (target.via === "asset") {
        const asset = await uploadAsset(file);
        // The server reads the bytes: an audio-only WebM or MP4 the browser called video gets the block it fits.
        const fits = inputForAssetKind(asset.kind);
        if (fits && fits !== target.type) node = createNode(fits, position);
        if (!node) continue;
        node.params = { ...node.params, asset };
      } else {
        if (file.size > MAX_TEXT_BYTES) {
          toast.error(`${file.name} is too large for a Text block`);
          continue;
        }
        node.params = { ...node.params, value: await file.text() };
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Upload failed");
      continue;
    }
    node.label = file.name.slice(0, 80);
    if (useDocStore.getState().flowId !== flowId) break;
    useDocStore.getState().addNode(node);
    added.push(node.id);
  }
  if (skipped) toast(`Skipped ${skipped} file(s) Flows cannot use here`);
  if (added.length && useDocStore.getState().flowId === flowId)
    useUiStore.getState().setSelection(added, []);
  return added;
}
