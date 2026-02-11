import {
  Background,
  BackgroundVariant,
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
import { Lock, LockOpen, Maximize2, Minus, Plus } from "lucide-react";
import { previewRecipe } from "./api";
import { RecipeGraphAuxNode, type RecipeGraphAuxNodeData } from "./components/recipe-graph-aux-node";
import { BlockSheet } from "./components/block-sheet";
import { RECIPE_FLOATING_ICON_BUTTON_CLASS } from "./components/recipe-floating-icon-button-class";
import { RecipeNode } from "./components/recipe-graph-node";
import { RecipeGraphSemanticEdge } from "./components/recipe-graph-semantic-edge";
import { DataEdge } from "./components/rf-ui/data-edge";
import { ConfigDialog } from "./dialogs/config-dialog";
import { ImportDialog } from "./dialogs/import-dialog";
import { ProcessorsDialog } from "./dialogs/processors-dialog";
import { useRecipeStudioStore } from "./stores/recipe-studio";
import type {
  RecipeNode as RecipeBuilderNode,
  RecipeNodeData,
  SamplerConfig,
} from "./types";
import { isCategoryConfig } from "./utils";
import { deriveDisplayGraph } from "./utils/graph/derive-display-graph";
import { importRecipePayload } from "./utils/import";
import { buildRecipePayload } from "./utils/payload";
import { buildDefaultSchemaTransform } from "./utils/processors";

const NODE_TYPES: NodeTypes = { builder: RecipeNode, aux: RecipeGraphAuxNode };
const EDGE_TYPES: EdgeTypes = { canvas: DataEdge, semantic: RecipeGraphSemanticEdge };

type LayoutControlsProps = {
  direction: "LR" | "TB";
  onLayout: () => void;
  onToggleDirection: () => void;
};

type ViewportControlsProps = {
  interactive: boolean;
  onToggleInteractive: () => void;
};

type InternalsSyncProps = {
  nodeIds: string[];
};

type StatusTone = "success" | "error";
type StatusMessage = {
  tone: StatusTone;
  text: string;
};

const STATUS_MESSAGE_CLASS: Record<StatusTone, string> = {
  success: "mt-2 text-xs text-emerald-600",
  error: "mt-2 text-xs text-rose-600",
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

function ViewportControls({
  interactive,
  onToggleInteractive,
}: ViewportControlsProps): ReactElement {
  const { zoomIn, zoomOut, fitView } = useReactFlow();

  const handleZoomIn = useCallback(() => {
    zoomIn({ duration: 150 });
  }, [zoomIn]);

  const handleZoomOut = useCallback(() => {
    zoomOut({ duration: 150 });
  }, [zoomOut]);

  const handleFitView = useCallback(() => {
    fitView({ duration: 250 });
  }, [fitView]);

  return (
    <Panel position="bottom-left" className="m-3 flex items-center gap-2">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomIn}
        aria-label="Zoom in"
      >
        <Plus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomOut}
        aria-label="Zoom out"
      >
        <Minus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleFitView}
        aria-label="Fit view"
      >
        <Maximize2 className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={onToggleInteractive}
        aria-label={interactive ? "Lock interaction" : "Unlock interaction"}
      >
        {interactive ? (
          <LockOpen className="size-4" />
        ) : (
          <Lock className="size-4" />
        )}
      </Button>
    </Panel>
  );
}

export function RecipeStudioPage(): ReactElement {
  const {
    nodes,
    edges,
    auxNodePositions,
    auxNodeSizes,
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
    loadRecipe,
    setLayoutDirection,
    applyLayout,
    setAuxNodePosition,
    setAuxNodeSize,
    syncAuxNodePositions,
    syncAuxNodeSizes,
  } = useRecipeStudioStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      auxNodePositions: state.auxNodePositions,
      auxNodeSizes: state.auxNodeSizes,
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
      loadRecipe: state.loadRecipe,
      setLayoutDirection: state.setLayoutDirection,
      applyLayout: state.applyLayout,
      setAuxNodePosition: state.setAuxNodePosition,
      setAuxNodeSize: state.setAuxNodeSize,
      syncAuxNodePositions: state.syncAuxNodePositions,
      syncAuxNodeSizes: state.syncAuxNodeSizes,
    })),
  );
  const [sheetContainer, setSheetContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const [previewLoading, setPreviewLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [interactive, setInteractive] = useState(true);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null);

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
      auxNodeSizes,
    });
  }, [auxNodePositions, auxNodeSizes, configs, edges, layoutDirection, nodes]);
  const displayNodeIds = useMemo(
    () => displayGraph.nodes.map((node) => node.id),
    [displayGraph.nodes],
  );
  useEffect(() => {
    syncAuxNodePositions(displayGraph.auxNodeIds, displayGraph.auxDefaults);
  }, [displayGraph.auxDefaults, displayGraph.auxNodeIds, syncAuxNodePositions]);
  useEffect(() => {
    syncAuxNodeSizes(displayGraph.auxNodeIds);
  }, [displayGraph.auxNodeIds, syncAuxNodeSizes]);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => {
      if (node.type !== "builder") {
        return;
      }
      selectConfig(node.id);
    },
    [selectConfig],
  );

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[]) => {
      for (const change of changes) {
        if (!("id" in change) || !change.id.startsWith("aux-")) {
          continue;
        }
        if (change.type === "position" && change.position) {
          setAuxNodePosition(change.id, change.position);
          continue;
        }
        if (
          change.type === "dimensions" &&
          change.dimensions &&
          change.dimensions.width > 0 &&
          change.dimensions.height > 0
        ) {
          setAuxNodeSize(change.id, {
            width: change.dimensions.width,
            height: change.dimensions.height,
          });
        }
      }
      const next = changes.filter(
        (change): change is NodeChange<RecipeBuilderNode> =>
          "id" in change && baseNodeIds.has(change.id),
      );
      if (next.length > 0) {
        onNodesChange(next);
      }
    },
    [baseNodeIds, onNodesChange, setAuxNodePosition, setAuxNodeSize],
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

  const configList = useMemo(() => Object.values(configs), [configs]);
  const config = activeConfigId ? configs[activeConfigId] : null;
  const categoryOptions = useMemo<SamplerConfig[]>(
    () => configList.filter(isCategoryConfig),
    [configList],
  );
  const modelConfigAliases = useMemo<string[]>(
    () => configList.filter((item) => item.kind === "model_config").map((item) => item.name),
    [configList],
  );
  const modelProviderOptions = useMemo<string[]>(
    () => configList.filter((item) => item.kind === "model_provider").map((item) => item.name),
    [configList],
  );
  const datetimeOptions = useMemo<string[]>(
    () =>
      configList
        .filter(
          (item) => item.kind === "sampler" && item.sampler_type === "datetime",
        )
        .map((item) => item.name),
    [configList],
  );

  const handleToggleDirection = useCallback(() => {
    setLayoutDirection(layoutDirection === "LR" ? "TB" : "LR");
  }, [layoutDirection, setLayoutDirection]);

  const toggleInteractive = useCallback(() => {
    setInteractive((value) => !value);
  }, []);

  const setSuccessStatus = useCallback((text: string) => {
    setStatusMessage({ tone: "success", text });
  }, []);

  const setErrorStatus = useCallback((text: string) => {
    setStatusMessage({ tone: "error", text });
  }, []);

  const buildPayload = useCallback(
    () => buildRecipePayload(configs, nodes, edges, processors),
    [configs, nodes, edges, processors],
  );

  const readPayload = useCallback(
    (fallbackError: string) => {
      const { payload, errors } = buildPayload();
      if (errors.length === 0) {
        return payload;
      }
      setErrorStatus(errors[0] ?? fallbackError);
      return null;
    },
    [buildPayload, setErrorStatus],
  );

  const handlePreview = async (): Promise<void> => {
    setPreviewLoading(true);
    setStatusMessage(null);
    const payload = readPayload("Fix config errors before preview.");
    if (!payload) {
      setPreviewLoading(false);
      return;
    }
    try {
      const result = await previewRecipe(payload);
      const rows = Array.isArray(result.dataset) ? result.dataset.length : 0;
      setSuccessStatus(`Preview ready (${rows} rows).`);
    } catch (error) {
      setErrorStatus(error instanceof Error ? error.message : "Preview failed.");
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleCopyRecipe = async (): Promise<void> => {
    setCopied(false);
    setStatusMessage(null);
    const payload = readPayload("Fix config errors before copy.");
    if (!payload) {
      return;
    }
    if (!navigator.clipboard) {
      setErrorStatus("Clipboard not available.");
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
      setSuccessStatus("Recipe copied to clipboard.");
    } catch (error) {
      setErrorStatus(error instanceof Error ? error.message : "Copy failed.");
    }
  };

  const handleImport = (value: string): string | null => {
    const result = importRecipePayload(value);
    if (result.errors.length > 0 || !result.snapshot) {
      return result.errors[0] ?? "Invalid payload.";
    }
    loadRecipe(result.snapshot);
    setSuccessStatus("Recipe imported.");
    return null;
  };

  const openProcessorsFromSheet = useCallback(() => {
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
              <h1 className="text-2xl font-semibold tracking-tight">Create Data Recipe</h1>
              <p className="text-sm text-muted-foreground">
                Design synthetic-data pipelines with Data Designer.
              </p>
              {statusMessage && (
                <p className={STATUS_MESSAGE_CLASS[statusMessage.tone]}>
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
            nodesDraggable={interactive}
            nodesConnectable={interactive}
            elementsSelectable={interactive}
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
                sheetView={sheetView}
                onViewChange={setSheetView}
                onAddSampler={addSamplerNode}
                onAddLlm={addLlmNode}
                onAddModelProvider={addModelProviderNode}
                onAddModelConfig={addModelConfigNode}
                onAddExpression={addExpressionNode}
                onOpenProcessors={openProcessorsFromSheet}
                copied={copied}
                onCopy={handleCopyRecipe}
                onImport={() => setImportOpen(true)}
              />
            </Panel>
            <ViewportControls
              interactive={interactive}
              onToggleInteractive={toggleInteractive}
            />
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
