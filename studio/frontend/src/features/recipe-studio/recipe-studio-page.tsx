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
import { PreviewDialog } from "./dialogs/preview-dialog";
import { ProcessorsDialog } from "./dialogs/processors-dialog";
import { useRecipeStudioActions } from "./hooks/use-recipe-studio-actions";
import { useRecipeStudioStore } from "./stores/recipe-studio";
import type {
  RecipeNode as RecipeBuilderNode,
  RecipeNodeData,
} from "./types";
import { deriveDisplayGraph } from "./utils/graph/derive-display-graph";
import { buildRecipePayload } from "./utils/payload";
import type { RecipePayload } from "./utils/payload/types";
import { buildDefaultSchemaTransform } from "./utils/processors";
import {
  applyAuxNodeChanges,
  filterEdgeChangesByIds,
  filterNodeChangesByIds,
} from "./utils/reactflow-changes";
import {
  buildDialogOptions,
  buildPreviewSummary,
} from "./utils/recipe-studio-view";

const NODE_TYPES: NodeTypes = { builder: RecipeNode, aux: RecipeGraphAuxNode };
const EDGE_TYPES: EdgeTypes = { canvas: DataEdge, semantic: RecipeGraphSemanticEdge };

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
    addSeedNode,
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
      addSeedNode: state.addSeedNode,
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
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [interactive, setInteractive] = useState(true);

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
      applyAuxNodeChanges(changes, { setAuxNodePosition, setAuxNodeSize });
      const next = filterNodeChangesByIds(
        changes as NodeChange<RecipeBuilderNode>[],
        baseNodeIds,
      );
      if (next.length) {
        onNodesChange(next);
      }
    },
    [baseNodeIds, onNodesChange, setAuxNodePosition, setAuxNodeSize],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      const next = filterEdgeChangesByIds(changes, baseEdgeIds);
      if (next.length) {
        onEdgesChange(next);
      }
    },
    [baseEdgeIds, onEdgesChange],
  );

  const configList = useMemo(() => Object.values(configs), [configs]);
  const config = activeConfigId ? configs[activeConfigId] : null;
  const dialogOptions = useMemo(
    () => buildDialogOptions(configList),
    [configList],
  );
  const previewSummary = useMemo(
    () => buildPreviewSummary(configList),
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
  const getCurrentPayloadFromStore = useCallback((): RecipePayload => {
    const state = useRecipeStudioStore.getState();
    return buildRecipePayload(
      state.configs,
      state.nodes,
      state.edges,
      state.processors,
    ).payload;
  }, []);
  const {
    workflowName,
    setWorkflowName,
    saveLoading,
    saveTone,
    savedAtLabel,
    copied,
    importOpen,
    setImportOpen,
    previewDialogOpen,
    setPreviewDialogOpen,
    previewRows,
    setPreviewRows,
    previewErrors,
    previewLoading,
    persistRecipe,
    openPreviewDialog,
    runPreview,
    copyRecipe,
    importRecipe,
  } = useRecipeStudioActions({
    recipeId,
    initialRecipeName,
    initialPayload,
    initialSavedAt,
    payloadResult,
    onPersistRecipe,
    resetRecipe,
    loadRecipe,
    getCurrentPayloadFromStore,
  });

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
            onPreview={openPreviewDialog}
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
                data: { path: "auto" },
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
                  onAddSeed={addSeedNode}
                  onAddLlm={addLlmNode}
                  onAddModelProvider={addModelProviderNode}
                  onAddModelConfig={addModelConfigNode}
                  onAddExpression={addExpressionNode}
                  onOpenProcessors={openProcessorsFromSheet}
                  copied={copied}
                  onCopy={copyRecipe}
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
        categoryOptions={dialogOptions.categoryOptions}
        modelConfigAliases={dialogOptions.modelConfigAliases}
        modelProviderOptions={dialogOptions.modelProviderOptions}
        datetimeOptions={dialogOptions.datetimeOptions}
        onUpdate={updateConfig}
        container={sheetContainer}
      />
      <ImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImport={importRecipe}
        container={sheetContainer}
      />
      <ProcessorsDialog
        open={processorsOpen}
        onOpenChange={setProcessorsOpen}
        processors={processors}
        onProcessorsChange={setProcessors}
        container={sheetContainer}
      />
      <PreviewDialog
        open={previewDialogOpen}
        onOpenChange={setPreviewDialogOpen}
        rows={previewRows}
        onRowsChange={setPreviewRows}
        loading={previewLoading}
        errors={previewErrors}
        summary={previewSummary}
        onPreview={() => {
          void runPreview();
        }}
        container={sheetContainer}
      />
    </div>
  );
}
