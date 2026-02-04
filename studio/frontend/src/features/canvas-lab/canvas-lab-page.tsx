import {
  Background,
  BackgroundVariant,
  type EdgeTypes,
  type Node,
  type NodeTypes,
  Panel,
  ReactFlow,
  useReactFlow,
} from "@xyflow/react";
import { type ReactElement, useCallback, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import "@xyflow/react/dist/style.css";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { EyeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { previewCanvas } from "./api";
import { BlockSheet } from "./components/block-sheet";
import { CanvasEdge } from "./components/canvas-edge";
import { CanvasNode } from "./components/canvas-node";
import { ConfigDialog } from "./dialogs/config-dialog";
import { ImportDialog } from "./dialogs/import-dialog";
import { useCanvasLabStore } from "./stores/canvas-lab";
import type { CanvasNodeData, SamplerConfig } from "./types";
import { isCategoryConfig } from "./utils";
import { importCanvasPayload } from "./utils/import";
import { buildCanvasPayload } from "./utils/payload";

const NODE_TYPES: NodeTypes = { builder: CanvasNode };
const EDGE_TYPES: EdgeTypes = { canvas: CanvasEdge };

type LayoutControlsProps = {
  direction: "LR" | "TB";
  onLayout: () => void;
  onToggleDirection: () => void;
};

function LayoutControls({
  direction,
  onLayout,
  onToggleDirection,
}: LayoutControlsProps): ReactElement {
  const { fitView } = useReactFlow();

  const handleLayout = useCallback(() => {
    onLayout();
    requestAnimationFrame(() => {
      fitView({ duration: 250 });
    });
  }, [fitView, onLayout]);

  return (
    <Panel position="top-left" className="m-3 flex items-center gap-2">
      <Button size="sm" variant="secondary" onClick={handleLayout}>
        Auto layout
      </Button>
      <Button size="sm" variant="outline" onClick={onToggleDirection}>
        {direction}
      </Button>
    </Panel>
  );
}

export function CanvasLabPage(): ReactElement {
  const {
    nodes,
    edges,
    configs,
    sheetView,
    activeConfigId,
    dialogOpen,
    layoutDirection,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addSamplerNode,
    addLlmNode,
    addExpressionNode,
    openConfig,
    updateConfig,
    isValidConnection,
    setSheetView,
    setDialogOpen,
    loadCanvas,
    setLayoutDirection,
    applyLayout,
  } = useCanvasLabStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      configs: state.configs,
      sheetView: state.sheetView,
      activeConfigId: state.activeConfigId,
      dialogOpen: state.dialogOpen,
      layoutDirection: state.layoutDirection,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addSamplerNode: state.addSamplerNode,
      addLlmNode: state.addLlmNode,
      addExpressionNode: state.addExpressionNode,
      openConfig: state.openConfig,
      updateConfig: state.updateConfig,
      isValidConnection: state.isValidConnection,
      setSheetView: state.setSheetView,
      setDialogOpen: state.setDialogOpen,
      loadCanvas: state.loadCanvas,
      setLayoutDirection: state.setLayoutDirection,
      applyLayout: state.applyLayout,
    })),
  );
  const [sheetContainer, setSheetContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const [previewLoading, setPreviewLoading] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{
    tone: "success" | "error";
    text: string;
  } | null>(null);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node<CanvasNodeData>) => {
      openConfig(node.id);
    },
    [openConfig],
  );

  const config = activeConfigId ? configs[activeConfigId] : null;
  const categoryOptions = useMemo<SamplerConfig[]>(
    () => Object.values(configs).filter(isCategoryConfig),
    [configs],
  );

  const handleToggleDirection = useCallback(() => {
    setLayoutDirection(layoutDirection === "LR" ? "TB" : "LR");
  }, [layoutDirection, setLayoutDirection]);

  const handlePreview = async (): Promise<void> => {
    setPreviewLoading(true);
    setStatusMessage(null);
    try {
      const { payload, errors } = buildCanvasPayload(configs, nodes, edges);
      if (errors.length > 0) {
        setStatusMessage({
          tone: "error",
          text: errors[0] ?? "Fix config errors before preview.",
        });
        return;
      }
      const result = await previewCanvas(payload);
      const rows = Array.isArray(result.dataset) ? result.dataset.length : 0;
      setStatusMessage({
        tone: "success",
        text: `Preview ready (${rows} rows).`,
      });
    } catch (error) {
      setStatusMessage({
        tone: "error",
        text: error instanceof Error ? error.message : "Preview failed.",
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleCopyRecipe = async (): Promise<void> => {
    setStatusMessage(null);
    const { payload, errors } = buildCanvasPayload(configs, nodes, edges);
    if (errors.length > 0) {
      setStatusMessage({
        tone: "error",
        text: errors[0] ?? "Fix config errors before copy.",
      });
      return;
    }
    if (!navigator.clipboard) {
      setStatusMessage({
        tone: "error",
        text: "Clipboard not available.",
      });
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setStatusMessage({
        tone: "success",
        text: "Recipe copied to clipboard.",
      });
    } catch (error) {
      setStatusMessage({
        tone: "error",
        text: error instanceof Error ? error.message : "Copy failed.",
      });
    }
  };

  const handleImport = (value: string): string | null => {
    const result = importCanvasPayload(value);
    if (result.errors.length > 0 || !result.snapshot) {
      return result.errors[0] ?? "Invalid payload.";
    }
    loadCanvas(result.snapshot);
    setStatusMessage({ tone: "success", text: "Recipe imported." });
    return null;
  };

  return (
    <div className="min-h-screen bg-background">
      <main className="w-full px-6 py-8">
        <div className="mb-6 flex flex-col gap-4">
          <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[1fr_auto] lg:items-center">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">
                Canvas Lab
              </h1>
              <p className="text-sm text-muted-foreground">
                Minimal React Flow canvas.
              </p>
              {statusMessage && (
                <p
                  className={
                    statusMessage.tone === "success"
                      ? "mt-2 text-xs text-emerald-600"
                      : "mt-2 text-xs text-rose-600"
                  }
                >
                  {statusMessage.text}
                </p>
              )}
            </div>
            <div className="flex items-center justify-start gap-2 lg:justify-end">
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={handleCopyRecipe}
                className="text-xs"
              >
                Copy recipe
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => setImportOpen(true)}
                className="text-xs"
              >
                Import
              </Button>
              <Button
                type="button"
                size="sm"
                onClick={handlePreview}
                disabled={previewLoading}
                className="gap-2 text-xs"
              >
                {previewLoading ? (
                  <Spinner className="size-3.5" />
                ) : (
                  <HugeiconsIcon icon={EyeIcon} className="size-3.5" />
                )}
                Preview
              </Button>
            </div>
          </div>
        </div>
        <div
          className="relative h-[75vh] w-full rounded-3xl border border-border/60 bg-white shadow-sm"
          ref={setSheetContainer}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            defaultEdgeOptions={{
              type: "canvas",
              style: { strokeWidth: 1.5, stroke: "#cbd5f5" },
            }}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={handleNodeClick}
            isValidConnection={isValidConnection}
            fitView={true}
            className="h-full w-full"
          >
            <LayoutControls
              direction={layoutDirection}
              onLayout={applyLayout}
              onToggleDirection={handleToggleDirection}
            />
            <Background
              variant={BackgroundVariant.Dots}
              gap={18}
              size={1}
              color="#d4d4d8"
            />
            <Panel position="top-right" className="m-3">
              <BlockSheet
                container={sheetContainer}
                view={sheetView}
                onViewChange={setSheetView}
                onAddSampler={addSamplerNode}
                onAddLlm={addLlmNode}
                onAddExpression={addExpressionNode}
              />
            </Panel>
          </ReactFlow>
        </div>
      </main>
      <ConfigDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        config={config}
        categoryOptions={categoryOptions}
        onUpdate={updateConfig}
      />
      <ImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImport={handleImport}
      />
    </div>
  );
}
