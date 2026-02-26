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
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  CookBookIcon,
  PlusSignIcon,
  TestTube01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
import { ExecutionsView } from "./components/executions/executions-view";
import { InternalsSync } from "./components/graph/internals-sync";
import { RecipeStudioHeader } from "./components/recipe-studio-header";
import { RecipeNode } from "./components/recipe-graph-node";
import { RecipeGraphSemanticEdge } from "./components/recipe-graph-semantic-edge";
import { DataEdge } from "./components/rf-ui/data-edge";
import { Button } from "@/components/ui/button";
import { ConfigDialog } from "./dialogs/config-dialog";
import { ImportDialog } from "./dialogs/import-dialog";
import { RunDialog } from "./dialogs/preview-dialog";
import { ProcessorsDialog } from "./dialogs/processors-dialog";
import { useRecipeStudioActions } from "./hooks/use-recipe-studio-actions";
import { useRecipeStudioStore } from "./stores/recipe-studio";
import type {
  RecipeNode as RecipeBuilderNode,
  RecipeNodeData,
} from "./types";
import { deriveDisplayGraph } from "./utils/graph/derive-display-graph";
import { getFitNodeIdsIgnoringNotes } from "./utils/graph/fit-view";
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
} from "./utils/recipe-studio-view";
import type { RecipeStudioView } from "./execution-types";

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
    llmAuxVisibility,
    configs,
    processors,
    sheetView,
    activeConfigId,
    dialogOpen,
    layoutDirection,
    fitViewTick,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addSamplerNode,
    addSeedNode,
    addLlmNode,
    addModelProviderNode,
    addModelConfigNode,
    addExpressionNode,
    addMarkdownNoteNode,
    selectConfig,
    openConfig,
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
  } = useRecipeStudioStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      auxNodePositions: state.auxNodePositions,
      llmAuxVisibility: state.llmAuxVisibility,
      configs: state.configs,
      processors: state.processors,
      sheetView: state.sheetView,
      activeConfigId: state.activeConfigId,
      dialogOpen: state.dialogOpen,
      layoutDirection: state.layoutDirection,
      fitViewTick: state.fitViewTick,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addSamplerNode: state.addSamplerNode,
      addSeedNode: state.addSeedNode,
      addLlmNode: state.addLlmNode,
      addModelProviderNode: state.addModelProviderNode,
      addModelConfigNode: state.addModelConfigNode,
      addExpressionNode: state.addExpressionNode,
      addMarkdownNoteNode: state.addMarkdownNoteNode,
      selectConfig: state.selectConfig,
      openConfig: state.openConfig,
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
    })),
  );
  const [sheetContainer, setSheetContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const [blockSheetOpen, setBlockSheetOpen] = useState(false);
  const [activeView, setActiveView] = useState<RecipeStudioView>("editor");
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [interactive, setInteractive] = useState(true);
  const [reactFlowInstance, setReactFlowInstance] = useState<
    ReactFlowInstance<Node<RecipeNodeData | RecipeGraphAuxNodeData>, Edge> | null
  >(null);
  const handleExecutionStart = useCallback(() => {
    setActiveView("executions");
  }, []);
  const handlePreviewSuccess = useCallback(() => {
    setActiveView("executions");
  }, []);

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
      llmAuxVisibility,
    });
  }, [
    auxNodePositions,
    configs,
    edges,
    layoutDirection,
    llmAuxVisibility,
    nodes,
  ]);
  const displayNodeIds = useMemo(
    () => displayGraph.nodes.map((node) => node.id),
    [displayGraph.nodes],
  );

  const handleNodeClick = useCallback(
    (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => {
      if (node.type !== "builder") {
        return;
      }
      selectConfig(node.id);
    },
    [selectConfig],
  );

  const handleNodeDoubleClick = useCallback(
    (_: unknown, node: Node<RecipeNodeData | RecipeGraphAuxNodeData>) => {
      if (node.type !== "builder") {
        return;
      }
      const nodeConfig = configs[node.id];
      if (nodeConfig?.kind === "markdown_note") {
        openConfig(node.id);
      }
    },
    [configs, openConfig],
  );

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[]) => {
      applyAuxNodeChanges(changes, { setAuxNodePosition });
      const next = filterNodeChangesByIds(
        changes as NodeChange<RecipeBuilderNode>[],
        baseNodeIds,
      );
      if (next.length) {
        onNodesChange(next);
      }
    },
    [baseNodeIds, onNodesChange, setAuxNodePosition],
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

  const handleToggleDirection = useCallback(() => {
    setLayoutDirection(layoutDirection === "LR" ? "TB" : "LR");
  }, [layoutDirection, setLayoutDirection]);

  const toggleInteractive = useCallback(() => {
    setInteractive((value) => !value);
  }, []);

  const payloadResult = useMemo(
    () =>
      buildRecipePayload(
        configs,
        nodes,
        edges,
        processors,
        layoutDirection,
        auxNodePositions,
      ),
    [auxNodePositions, configs, edges, layoutDirection, nodes, processors],
  );
  const getCurrentPayloadFromStore = useCallback((): RecipePayload => {
    const state = useRecipeStudioStore.getState();
    return buildRecipePayload(
      state.configs,
      state.nodes,
      state.edges,
      state.processors,
      state.layoutDirection,
      state.auxNodePositions,
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
    runDialogOpen,
    runDialogKind,
    setRunDialogKind,
    setRunDialogOpen,
    previewRows,
    fullRows,
    setPreviewRows,
    setFullRows,
    runErrors,
    runSettings,
    setRunSettings,
    previewLoading,
    fullLoading,
    currentSignature,
    executions,
    selectedExecutionId,
    setSelectedExecutionId,
    persistRecipe,
    openRunDialog,
    runFromDialog,
    validateFromDialog,
    validateLoading,
    validateResult,
    cancelExecution,
    loadExecutionDatasetPage,
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
    onExecutionStart: handleExecutionStart,
    onPreviewSuccess: handlePreviewSuccess,
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

  const openRootBlockSheet = useCallback(() => {
    setSheetView("root");
    setBlockSheetOpen(true);
  }, [setSheetView]);
  const runDialogRows = runDialogKind === "preview" ? previewRows : fullRows;
  const runDialogLoading =
    runDialogKind === "preview" ? previewLoading : fullLoading;

  useEffect(() => {
    if (!reactFlowInstance || activeView !== "editor" || fitViewTick === 0) {
      return;
    }
    let frame2 = 0;
    const frame1 = window.requestAnimationFrame(() => {
      frame2 = window.requestAnimationFrame(() => {
        reactFlowInstance.fitView({
          duration: 250,
          nodes: getFitNodeIdsIgnoringNotes(reactFlowInstance.getNodes()),
        });
      });
    });
    return () => {
      window.cancelAnimationFrame(frame1);
      if (frame2) {
        window.cancelAnimationFrame(frame2);
      }
    };
  }, [activeView, fitViewTick, reactFlowInstance]);

  return (
    <div className="min-h-screen bg-background">
      <main className="w-full px-6 py-8">
        <div
          className="relative w-full overflow-hidden rounded-2xl corner-squircle border"
          ref={setSheetContainer}
        >
          <RecipeStudioHeader
            activeView={activeView}
            saveLoading={saveLoading}
            saveTone={saveTone}
            savedAtLabel={savedAtLabel}
            workflowName={workflowName}
            onWorkflowNameChange={setWorkflowName}
            onViewChange={setActiveView}
            onSaveRecipe={() => {
              void persistRecipe();
            }}
          />
          <div className="h-[75vh] w-full rounded-t-none">
            {activeView === "editor" ? (
              <ReactFlow<Node<RecipeNodeData | RecipeGraphAuxNodeData>, Edge>
                onInit={setReactFlowInstance}
                nodes={displayGraph.nodes}
                edges={displayGraph.edges}
                nodeTypes={NODE_TYPES}
                edgeTypes={EDGE_TYPES}
                defaultEdgeOptions={{
                  type: "canvas",
                  data: { path: "smoothstep" },
                }}
                onNodesChange={handleNodesChange}
                onEdgesChange={handleEdgesChange}
                onConnect={onConnect}
                onNodeClick={handleNodeClick}
                onNodeDoubleClick={handleNodeDoubleClick}
                isValidConnection={isValidConnection}
                nodesDraggable={interactive}
                nodesConnectable={interactive}
                elementsSelectable={interactive}
                fitView={false}
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
                {nodes.length === 0 && (
                  <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center p-4">
                    <button
                      type="button"
                      onClick={openRootBlockSheet}
                      className="pointer-events-auto corner-squircle flex min-h-36 w-full max-w-md flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-border/70 bg-background/75 px-6 py-6 text-center backdrop-blur-[1px] transition hover:border-primary/60 hover:bg-background"
                    >
                      <div className="flex size-12 items-center justify-center corner-squircle rounded-xl border border-border/70 bg-muted/40">
                        <HugeiconsIcon
                          icon={PlusSignIcon}
                          className="size-6 text-muted-foreground"
                        />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-foreground">
                          Add your first block
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Click to open block library.
                        </p>
                      </div>
                    </button>
                  </div>
                )}
                <Panel position="top-right" className="m-3">
                  <BlockSheet
                    container={sheetContainer}
                    sheetView={sheetView}
                    onViewChange={setSheetView}
                    open={blockSheetOpen}
                    onOpenChange={setBlockSheetOpen}
                    onAddSampler={addSamplerNode}
                    onAddSeed={addSeedNode}
                    onAddLlm={addLlmNode}
                    onAddModelProvider={addModelProviderNode}
                    onAddModelConfig={addModelConfigNode}
                    onAddExpression={addExpressionNode}
                    onAddMarkdownNote={addMarkdownNoteNode}
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
                <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center">
                  <div className="pointer-events-auto flex items-center gap-2">
                    <Button
                      type="button"
                      className="h-11 px-5"
                      onClick={() => openRunDialog(runDialogKind)}
                      disabled={previewLoading || fullLoading}
                    >
                      <HugeiconsIcon icon={CookBookIcon} className="size-4" />
                      {previewLoading || fullLoading ? "Running..." : "Run"}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      className="h-11 px-5"
                      onClick={() => {
                        openRunDialog(runDialogKind);
                        void validateFromDialog();
                      }}
                      disabled={validateLoading}
                    >
                      <HugeiconsIcon icon={TestTube01Icon} className="size-4" />
                      {validateLoading ? "Validating..." : "Validate"}
                    </Button>
                  </div>
                </div>
              </ReactFlow>
            ) : (
              <ExecutionsView
                executions={executions}
                selectedExecutionId={selectedExecutionId}
                currentSignature={currentSignature}
                onSelectExecution={setSelectedExecutionId}
                onCancelExecution={(executionId) => {
                  void cancelExecution(executionId);
                }}
                onLoadDatasetPage={(executionId, page) => {
                  void loadExecutionDatasetPage(executionId, page);
                }}
              />
            )}
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
      <RunDialog
        open={runDialogOpen}
        onOpenChange={setRunDialogOpen}
        kind={runDialogKind}
        onKindChange={setRunDialogKind}
        rows={runDialogRows}
        onRowsChange={(rows) => {
          if (runDialogKind === "preview") {
            setPreviewRows(rows);
            return;
          }
          setFullRows(rows);
        }}
        settings={runSettings}
        onSettingsChange={setRunSettings}
        loading={runDialogLoading}
        validateLoading={validateLoading}
        validateResult={validateResult}
        errors={runErrors}
        onValidate={() => {
          void validateFromDialog();
        }}
        onRun={() => {
          void runFromDialog();
        }}
        container={sheetContainer}
      />
    </div>
  );
}
