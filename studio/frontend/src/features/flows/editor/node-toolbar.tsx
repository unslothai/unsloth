// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { NodeToolbar, Position } from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { quickModelParam, toolbarNodeId } from "./canvas-rules";
import { DiceIcon, Download01Icon, PlayIcon } from "./icon-data";
import { downloadNodeOutput, rerollNode, runToNode } from "./node-actions";
import { ParamField } from "./param-fields";

const BUTTON =
  "flex items-center gap-1 rounded-md px-2 py-1 text-[11.5px] text-zinc-200 hover:bg-white/10";

function SelectedNodeToolbarImpl({ readOnly }: { readOnly: boolean }) {
  const nodeId = useUiStore((s) =>
    toolbarNodeId(
      s.selectedNodes,
      (id) => !!useDocStore.getState().graph.nodes[id],
      readOnly,
    ),
  );
  const node = useDocStore((s) => (nodeId ? s.graph.nodes[nodeId] : undefined));
  const spec = useCatalogStore((s) => (node ? s.specs[node.type] : undefined));
  const hasOutput = useRuntimeStore(
    (s) => !!(nodeId && s.nodes[nodeId]?.outputs),
  );
  if (!nodeId || !node || !spec) return null;
  const modelParam = quickModelParam(spec);
  return (
    <NodeToolbar
      nodeId={nodeId}
      isVisible={true}
      position={Position.Top}
      offset={8}
      className="nodrag nopan nowheel flex items-center gap-1 rounded-lg border border-white/10 bg-zinc-900/95 p-1 shadow-lg shadow-black/40"
    >
      <button
        type="button"
        className={BUTTON}
        title="Run only what this block needs"
        data-testid="node-toolbar-run"
        onClick={() => void runToNode(nodeId)}
      >
        <HugeiconsIcon icon={PlayIcon} size={12} /> Run
      </button>
      <button
        type="button"
        className={BUTTON}
        title="Run this block again, ignoring the cache (new seed when it has one)"
        data-testid="node-toolbar-reroll"
        onClick={() => void rerollNode(nodeId)}
      >
        <HugeiconsIcon icon={DiceIcon} size={12} /> Regenerate
      </button>
      {hasOutput && (
        <button
          type="button"
          className={BUTTON}
          title="Download this block's last output"
          data-testid="node-toolbar-download"
          onClick={() => void downloadNodeOutput(nodeId)}
        >
          <HugeiconsIcon icon={Download01Icon} size={12} /> Download
        </button>
      )}
      {modelParam && (
        <div className="w-56" data-testid="node-toolbar-model">
          <ParamField
            spec={modelParam}
            value={node.params[modelParam.id]}
            onChange={(value) =>
              useDocStore
                .getState()
                .updateParams(
                  nodeId,
                  { [modelParam.id]: value },
                  `param:${nodeId}:${modelParam.id}`,
                )
            }
          />
        </div>
      )}
    </NodeToolbar>
  );
}

export const SelectedNodeToolbar = memo(SelectedNodeToolbarImpl);
