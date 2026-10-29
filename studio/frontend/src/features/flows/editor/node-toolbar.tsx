// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { NodeToolbar, Position, useReactFlow, useStore } from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { quickModelParam, toolbarNodeId } from "./canvas-rules";
import { Delete02Icon, DiceIcon, Download01Icon, PlayIcon } from "./icon-data";
import { downloadNodeOutput, rerollNode, runToNode } from "./node-actions";
import { ParamField } from "./param-fields";

// Room the toolbar needs above a card: one row, or the row plus the model picker.
const TOOLBAR_ROOM = 48;
const TOOLBAR_ROOM_MODEL = 128;

/** Non-scaling by design: NodeToolbar sits outside the zoomed viewport, so it keeps its screen
 *  size at any zoom while staying anchored to the card. */
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
  const { deleteElements } = useReactFlow();
  const room = quickModelParam(spec) ? TOOLBAR_ROOM_MODEL : TOOLBAR_ROOM;
  // Above the card, or below it when the card's top is too close to the canvas edge to fit it.
  const below = useStore((s) => {
    const internal = nodeId ? s.nodeLookup.get(nodeId) : undefined;
    return (
      !!internal &&
      internal.internals.positionAbsolute.y * s.transform[2] + s.transform[1] <
        room
    );
  });
  if (!nodeId || !node || !spec) return null;
  const modelParam = quickModelParam(spec);
  return (
    <NodeToolbar
      nodeId={nodeId}
      isVisible={true}
      position={below ? Position.Bottom : Position.Top}
      offset={8}
      className="nodrag nopan nowheel flex items-center gap-1 rounded-lg border border-border bg-popover p-1 shadow-[var(--popup-shadow)]"
    >
      <Button
        type="button"
        variant="ghost"
        size="xs"
        title="Run only what this block needs"
        data-testid="node-toolbar-run"
        onClick={() => void runToNode(nodeId)}
      >
        <HugeiconsIcon icon={PlayIcon} size={12} /> Run
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="xs"
        title="Run this block again, ignoring the cache (new seed when it has one)"
        data-testid="node-toolbar-reroll"
        onClick={() => void rerollNode(nodeId)}
      >
        <HugeiconsIcon icon={DiceIcon} size={12} /> Regenerate
      </Button>
      {hasOutput && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          title="Download this block's last output"
          data-testid="node-toolbar-download"
          onClick={() => void downloadNodeOutput(nodeId)}
        >
          <HugeiconsIcon icon={Download01Icon} size={12} /> Download
        </Button>
      )}
      <Button
        type="button"
        variant="ghost"
        size="xs"
        title="Delete this block (Delete)"
        aria-label="Delete block"
        data-testid="node-toolbar-delete"
        className="text-muted-foreground hover:text-destructive"
        onClick={() => void deleteElements({ nodes: [{ id: nodeId }] })}
      >
        <HugeiconsIcon icon={Delete02Icon} size={12} />
      </Button>
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
