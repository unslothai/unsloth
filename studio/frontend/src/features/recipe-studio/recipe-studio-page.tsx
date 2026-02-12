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
import { previewRecipe } from "./api";
import { RecipeGraphAuxNode, type RecipeGraphAuxNodeData } from "./components/recipe-graph-aux-node";
import { BlockSheet } from "./components/block-sheet";
import { LayoutControls } from "./components/controls/layout-controls";
import { ViewportControls } from "./components/controls/viewport-controls";
import { InternalsSync } from "./components/graph/internals-sync";
import { RecipeStudioHeader } from "./components/recipe-studio-header";
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
import type { RecipePayload } from "./utils/payload/types";
import { buildDefaultSchemaTransform } from "./utils/processors";

const NODE_TYPES: NodeTypes = { builder: RecipeNode, aux: RecipeGraphAuxNode };
const EDGE_TYPES: EdgeTypes = { canvas: DataEdge, semantic: RecipeGraphSemanticEdge };

type StatusTone = "success" | "error";

export type PersistRecipeInput = {
  id: string | null;
  name: string;
  payload: RecipePayload;
};

export type PersistRecipeResult = {
  id: string;
  updatedAt: number;
};

export type RecipeStudioPageProps = {
  recipeId: string;
  initialRecipeName: string;
  initialPayload: RecipePayload;
  initialSavedAt: number;
  onPersistRecipe: (input: PersistRecipeInput) => Promise<PersistRecipeResult>;
};

function buildSignature(name: string, payload: RecipePayload): string {
  return JSON.stringify({ name, payload });
}

function normalizeWorkflowName(value: string): string {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : "Unnamed";
}

function formatSavedLabel(savedAt: number | null): string {
  if (!savedAt) {
    return "Not saved yet";
  }
  const time = new Date(savedAt).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
  return `Saved ${time}`;
}

export function RecipeStudioPage({
  recipeId,
  initialRecipeName,
  initialPayload,
  initialSavedAt,
  onPersistRecipe,
}: RecipeStudioPageProps): ReactElement {
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
    resetRecipe,
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
      resetRecipe: state.resetRecipe,
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
  const [saveLoading, setSaveLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [interactive, setInteractive] = useState(true);
  const [workflowName, setWorkflowName] = useState("Unnamed");
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [savedSignature, setSavedSignature] = useState<string>("");

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

  const payloadResult = useMemo(
    () => buildRecipePayload(configs, nodes, edges, processors),
    [configs, edges, nodes, processors],
  );
  const currentPayload = payloadResult.payload;
  const normalizedWorkflowName = useMemo(
    () => normalizeWorkflowName(workflowName),
    [workflowName],
  );
  const currentSignature = useMemo(
    () => buildSignature(normalizedWorkflowName, currentPayload),
    [currentPayload, normalizedWorkflowName],
  );
  const isDirty = savedSignature.length > 0 && currentSignature !== savedSignature;
  const saveTone: StatusTone =
    !isDirty && Boolean(lastSavedAt) ? "success" : "error";
  const savedAtLabel = formatSavedLabel(lastSavedAt);

  useEffect(() => {
    const nextName = normalizeWorkflowName(initialRecipeName);
    resetRecipe();
    setWorkflowName(nextName);
    setLastSavedAt(initialSavedAt);
    setCopied(false);

    const parsed = importRecipePayload(JSON.stringify(initialPayload));
    if (parsed.snapshot) {
      loadRecipe(parsed.snapshot);
    } else {
      console.error("Failed to load recipe payload.", parsed.errors);
    }

    const state = useRecipeStudioStore.getState();
    const { payload } = buildRecipePayload(
      state.configs,
      state.nodes,
      state.edges,
      state.processors,
    );
    setSavedSignature(buildSignature(nextName, payload));
  }, [
    initialPayload,
    initialRecipeName,
    initialSavedAt,
    loadRecipe,
    recipeId,
    resetRecipe,
  ]);

  const persistRecipe = useCallback(async (): Promise<void> => {
    if (saveLoading) {
      return;
    }
    const nextName = normalizeWorkflowName(workflowName);
    if (nextName !== workflowName) {
      setWorkflowName(nextName);
    }
    setSaveLoading(true);
    try {
      const result = await onPersistRecipe({
        id: recipeId,
        name: nextName,
        payload: currentPayload,
      });
      setLastSavedAt(result.updatedAt);
      setSavedSignature(buildSignature(nextName, currentPayload));
    } catch (error) {
      console.error("Save recipe failed:", error);
    } finally {
      setSaveLoading(false);
    }
  }, [
    currentPayload,
    onPersistRecipe,
    recipeId,
    saveLoading,
    workflowName,
  ]);

  useEffect(() => {
    if (!isDirty || saveLoading) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      void persistRecipe();
    }, 800);
    return () => window.clearTimeout(timeoutId);
  }, [isDirty, persistRecipe, saveLoading]);

  const readPayload = useCallback(
    () => {
      if (payloadResult.errors.length === 0) {
        return payloadResult.payload;
      }
      return null;
    },
    [payloadResult.errors.length, payloadResult.payload],
  );

  const handlePreview = async (): Promise<void> => {
    setPreviewLoading(true);
    const payload = readPayload();
    if (!payload) {
      setPreviewLoading(false);
      return;
    }
    try {
      await previewRecipe(payload);
    } catch (error) {
      console.error("Preview failed:", error);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleCopyRecipe = async (): Promise<void> => {
    setCopied(false);
    const payload = readPayload();
    if (!payload) {
      return;
    }
    if (!navigator.clipboard) {
      console.error("Clipboard not available.");
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (error) {
      console.error("Copy failed:", error);
    }
  };

  const handleImport = (value: string): string | null => {
    const result = importRecipePayload(value);
    if (result.errors.length > 0 || !result.snapshot) {
      return result.errors[0] ?? "Invalid payload.";
    }
    loadRecipe(result.snapshot);
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
        <div
          className="relative w-full overflow-hidden rounded-2xl corner-squircle border"
          ref={setSheetContainer}
        >
          <RecipeStudioHeader
            previewLoading={previewLoading}
            saveLoading={saveLoading}
            saveTone={saveTone}
            savedAtLabel={savedAtLabel}
            workflowName={workflowName}
            onWorkflowNameChange={setWorkflowName}
            onPreview={handlePreview}
            onSaveRecipe={() => {
              void persistRecipe();
            }}
          />
          <div className="h-[75vh] w-full rounded-t-none">
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
              className="h-full w-full rounded-t-none"
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
