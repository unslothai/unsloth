import {
  Background,
  BackgroundVariant,
  Controls,
  type Edge,
  type EdgeChange,
  type EdgeTypes,
  type Node,
  type NodeChange,
  type NodeTypes,
  Panel,
  ReactFlow,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import "@xyflow/react/dist/style.css";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { EyeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { previewCanvas } from "./api";
import { CanvasAuxNode, type CanvasAuxNodeData } from "./components/canvas-aux-node";
import { BlockSheet } from "./components/block-sheet";
import { CanvasNode } from "./components/canvas-node";
import { CanvasSemanticEdge } from "./components/canvas-semantic-edge";
import { DataEdge } from "./components/rf-ui/data-edge";
import { ConfigDialog } from "./dialogs/config-dialog";
import { ImportDialog } from "./dialogs/import-dialog";
import { ProcessorsDialog } from "./dialogs/processors-dialog";
import { useCanvasLabStore } from "./stores/canvas-lab";
import type {
  CanvasNode as CanvasBuilderNode,
  CanvasNodeData,
  SamplerConfig,
} from "./types";
import { isCategoryConfig } from "./utils";
import { deriveDisplayGraph } from "./utils/graph/derive-display-graph";
import { importCanvasPayload } from "./utils/import";
import { buildCanvasPayload } from "./utils/payload";
import { buildDefaultSchemaTransform } from "./utils/processors";

const NODE_TYPES: NodeTypes = { builder: CanvasNode, aux: CanvasAuxNode };
const EDGE_TYPES: EdgeTypes = { canvas: DataEdge, semantic: CanvasSemanticEdge };

type LayoutControlsProps = {
  direction: "LR" | "TB";
  onLayout: () => void;
  onToggleDirection: () => void;
};

type InternalsSyncProps = {
  nodeIds: string[];
};

function InternalsSync({ nodeIds }: InternalsSyncProps): null {
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    if (nodeIds.length === 0) {
      return;
    }
    requestAnimationFrame(() => {
      updateNodeInternals(nodeIds);
      requestAnimationFrame(() => {
        updateNodeInternals(nodeIds);
      });
    });
  }, [nodeIds, updateNodeInternals]);

  return null;
}

function LayoutControls({
  direction,
  onLayout,
  onToggleDirection,
}: LayoutControlsProps): ReactElement {
  const { fitView, getNodes } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  const refreshNodeInternals = useCallback(() => {
    const nodeIds = getNodes().map((node) => node.id);
    if (nodeIds.length > 0) {
      updateNodeInternals(nodeIds);
    }
  }, [getNodes, updateNodeInternals]);

  const handleLayout = useCallback(() => {
    onLayout();
    requestAnimationFrame(() => {
      refreshNodeInternals();
      requestAnimationFrame(() => {
        fitView({ duration: 250 });
      });
    });
  }, [fitView, onLayout, refreshNodeInternals]);

  const handleToggleDirection = useCallback(() => {
    onToggleDirection();
    requestAnimationFrame(() => {
      refreshNodeInternals();
      requestAnimationFrame(() => {
        refreshNodeInternals();
      });
    });
  }, [onToggleDirection, refreshNodeInternals]);

  return (
    <Panel position="top-left" className="m-3 flex items-center gap-2">
      <Button size="sm" className="corner-squircle" variant="secondary" onClick={handleLayout}>
        Auto layout
      </Button>
      <Button size="sm" className="corner-squircle" variant="outline" onClick={handleToggleDirection}>
        {direction}
      </Button>
    </Panel>
  );
}

export function CanvasLabPage(): ReactElement {
  const {
    nodes,
    edges,
    auxNodePositions,
    configs,
    processors,
    sheetView,
    activeConfigId,
    dialogOpen,
    layoutDirection,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addSamplerNode,
    addLlmNode,
    addModelProviderNode,
    addModelConfigNode,
    addExpressionNode,
    selectConfig,
    updateConfig,
    isValidConnection,
    setSheetView,
    setProcessors,
    setDialogOpen,
    loadCanvas,
    setLayoutDirection,
    applyLayout,
    setAuxNodePosition,
    syncAuxNodePositions,
  } = useCanvasLabStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      auxNodePositions: state.auxNodePositions,
      configs: state.configs,
      processors: state.processors,
      sheetView: state.sheetView,
      activeConfigId: state.activeConfigId,
      dialogOpen: state.dialogOpen,
      layoutDirection: state.layoutDirection,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addSamplerNode: state.addSamplerNode,
      addLlmNode: state.addLlmNode,
      addModelProviderNode: state.addModelProviderNode,
      addModelConfigNode: state.addModelConfigNode,
      addExpressionNode: state.addExpressionNode,
      selectConfig: state.selectConfig,
      updateConfig: state.updateConfig,
      isValidConnection: state.isValidConnection,
      setSheetView: state.setSheetView,
      setProcessors: state.setProcessors,
      setDialogOpen: state.setDialogOpen,
      loadCanvas: state.loadCanvas,
      setLayoutDirection: state.setLayoutDirection,
      applyLayout: state.applyLayout,
      setAuxNodePosition: state.setAuxNodePosition,
      syncAuxNodePositions: state.syncAuxNodePositions,
    })),
  );
  const [sheetContainer, setSheetContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const [previewLoading, setPreviewLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{
    tone: "success" | "error";
    text: string;
  } | null>(null);

  const baseNodeIds = useMemo(
    () => new Set(nodes.map((node) => node.id)),
    [nodes],
  );
  const baseEdgeIds = useMemo(
    () => new Set(edges.map((edge) => edge.id)),
    [edges],
  );

  const displayGraph = useMemo(() => {
    return deriveDisplayGraph({
      nodes,
      edges,
      configs,
      layoutDirection,
      auxNodePositions,
    });
  }, [auxNodePositions, configs, edges, layoutDirection, nodes]);
  const displayNodeIds = useMemo(
    () => displayGraph.nodes.map((node) => node.id),
    [displayGraph.nodes],
  );
  useEffect(() => {
    syncAuxNodePositions(displayGraph.auxNodeIds, displayGraph.auxDefaults);
  }, [displayGraph.auxDefaults, displayGraph.auxNodeIds, syncAuxNodePositions]);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node<CanvasNodeData | CanvasAuxNodeData>) => {
      if (node.type !== "builder") {
        return;
      }
      selectConfig(node.id);
    },
    [selectConfig],
  );

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<CanvasNodeData | CanvasAuxNodeData>>[]) => {
      for (const change of changes) {
        if (
          !("id" in change) ||
          change.type !== "position" ||
          !change.id.startsWith("aux-") ||
          !change.position
        ) {
          continue;
        }
        setAuxNodePosition(change.id, change.position);
      }
      const next = changes.filter(
        (change): change is NodeChange<CanvasBuilderNode> =>
          "id" in change && baseNodeIds.has(change.id),
      );
      if (next.length > 0) {
        onNodesChange(next);
      }
    },
    [baseNodeIds, onNodesChange, setAuxNodePosition],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      const next = changes.filter(
        (change): change is EdgeChange<Edge> =>
          "id" in change && baseEdgeIds.has(change.id),
      );
      if (next.length > 0) {
        onEdgesChange(next);
      }
    },
    [baseEdgeIds, onEdgesChange],
  );

  const config = activeConfigId ? configs[activeConfigId] : null;
  const categoryOptions = useMemo<SamplerConfig[]>(
    () => Object.values(configs).filter(isCategoryConfig),
    [configs],
  );
  const modelConfigAliases = useMemo<string[]>(
    () =>
      Object.values(configs)
        .filter((item) => item.kind === "model_config")
        .map((item) => item.name),
    [configs],
  );
  const modelProviderOptions = useMemo<string[]>(
    () =>
      Object.values(configs)
        .filter((item) => item.kind === "model_provider")
        .map((item) => item.name),
    [configs],
  );
  const datetimeOptions = useMemo<string[]>(
    () =>
      Object.values(configs)
        .filter(
          (item) => item.kind === "sampler" && item.sampler_type === "datetime",
        )
        .map((item) => item.name),
    [configs],
  );

  const handleToggleDirection = useCallback(() => {
    setLayoutDirection(layoutDirection === "LR" ? "TB" : "LR");
  }, [layoutDirection, setLayoutDirection]);

  const handlePreview = async (): Promise<void> => {
    setPreviewLoading(true);
    setStatusMessage(null);
    try {
      const { payload, errors } = buildCanvasPayload(
        configs,
        nodes,
        edges,
        processors,
      );
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
    setCopied(false);
    setStatusMessage(null);
    const { payload, errors } = buildCanvasPayload(
      configs,
      nodes,
      edges,
      processors,
    );
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
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
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

  const handleOpenProcessorsFromSheet = useCallback(() => {
    if (
      !processors.some(
        (processor) => processor.processor_type === "schema_transform",
      )
    ) {
      setProcessors([...processors, buildDefaultSchemaTransform()]);
    }
    setProcessorsOpen(true);
  }, [processors, setProcessors]);

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
          className="relative h-[75vh] w-full rounded-2xl corner-squircle border "
          ref={setSheetContainer}
        >
          <ReactFlow
            nodes={displayGraph.nodes}
            edges={displayGraph.edges}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            defaultEdgeOptions={{
              type: "canvas",
              data: { key: "name", path: "auto" },
              style: { strokeWidth: 1.5, stroke: "var(--border)" },
            }}
            onNodesChange={handleNodesChange}
            onEdgesChange={handleEdgesChange}
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
            <InternalsSync nodeIds={displayNodeIds} />
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
                onAddModelProvider={addModelProviderNode}
                onAddModelConfig={addModelConfigNode}
                onAddExpression={addExpressionNode}
                onOpenProcessors={handleOpenProcessorsFromSheet}
                copied={copied}
                onCopy={handleCopyRecipe}
                onImport={() => setImportOpen(true)}
              />
            </Panel>
            <Controls position="bottom-left" />
          </ReactFlow>
        </div>
      </main>
      <ConfigDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        config={config}
        categoryOptions={categoryOptions}
        modelConfigAliases={modelConfigAliases}
        modelProviderOptions={modelProviderOptions}
        datetimeOptions={datetimeOptions}
        onUpdate={updateConfig}
        container={sheetContainer}
      />
      <ImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImport={handleImport}
        container={sheetContainer}
      />
      <ProcessorsDialog
        open={processorsOpen}
        onOpenChange={setProcessorsOpen}
        processors={processors}
        onProcessorsChange={setProcessors}
        container={sheetContainer}
      />
    </div>
  );
}
