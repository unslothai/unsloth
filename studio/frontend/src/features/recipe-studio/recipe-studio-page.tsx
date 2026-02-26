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
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  CookBookIcon,
  DiceFaces03Icon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  PlusSignIcon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  TestTube01Icon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type DragEvent as ReactDragEvent,
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import "@xyflow/react/dist/style.css";
import { RecipeGraphAuxNode, type RecipeGraphAuxNodeData } from "./components/recipe-graph-aux-node";
import {
  BlockSheet,
  RECIPE_BLOCK_DND_MIME,
  type RecipeBlockDragPayload,
} from "./components/block-sheet";
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
  LlmType,
  NodeConfig,
  RecipeNode as RecipeBuilderNode,
  RecipeNodeData,
  SamplerType,
} from "./types";
import type { SeedBlockType } from "./blocks/registry";
import { deriveDisplayGraph } from "./utils/graph/derive-display-graph";
import { getFitNodeIdsIgnoringNotes } from "./utils/graph/fit-view";
import {
  deriveGraphRuntimeVisualState,
  pickLatestActiveExecution,
} from "./utils/graph/runtime-visual-state";
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
type IconType = typeof CodeIcon;
const SUPPORTED_DRAG_KINDS: RecipeBlockDragPayload["kind"][] = [
  "sampler",
  "seed",
  "llm",
  "expression",
  "note",
];

const SAMPLER_ICONS: Record<SamplerType, IconType> = {
  category: Tag01Icon,
  subcategory: TagsIcon,
  uniform: EqualSignIcon,
  gaussian: Parabola02Icon,
  bernoulli: EqualSignIcon,
  datetime: Clock01Icon,
  timedelta: Clock01Icon,
  uuid: FingerPrintIcon,
  person: UserAccountIcon,
  person_from_faker: UserAccountIcon,
};

const LLM_ICONS: Record<LlmType, IconType> = {
  text: PencilEdit02Icon,
  structured: CodeIcon,
  code: CodeSimpleIcon,
  judge: BalanceScaleIcon,
};

function resolveExecutionColumnIcon(config: NodeConfig | null): IconType {
  if (!config) {
    return DiceFaces03Icon;
  }
  if (config.kind === "sampler") {
    return SAMPLER_ICONS[config.sampler_type];
  }
  if (config.kind === "llm") {
    return LLM_ICONS[config.llm_type];
  }
  if (config.kind === "expression") {
    return FunctionIcon;
  }
  if (config.kind === "seed") {
    return Plant01Icon;
  }
  if (config.kind === "model_provider") {
    return Shield02Icon;
  }
  if (config.kind === "model_config") {
    return Plant01Icon;
  }
  return PencilEdit02Icon;
}

function parseRecipeBlockDragPayload(raw: string): RecipeBlockDragPayload | null {
  try {
    const parsed = JSON.parse(raw) as {
      kind?: RecipeBlockDragPayload["kind"];
      type?: RecipeBlockDragPayload["type"];
    };
    if (
      !parsed.kind ||
      !parsed.type ||
      !SUPPORTED_DRAG_KINDS.includes(parsed.kind)
    ) {
      return null;
    }
    return {
      kind: parsed.kind,
      type: parsed.type,
    };
  } catch {
    return null;
  }
}

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
    setExecutionLocked,
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
      setExecutionLocked: state.setExecutionLocked,
    })),
  );
  const [sheetContainer, setSheetContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const flowContainerRef = useRef<HTMLDivElement | null>(null);
  const [blockSheetOpen, setBlockSheetOpen] = useState(false);
  const [activeView, setActiveView] = useState<RecipeStudioView>("editor");
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [interactive, setInteractive] = useState(true);
  const [reactFlowInstance, setReactFlowInstance] = useState<
    ReactFlowInstance<Node<RecipeNodeData | RecipeGraphAuxNodeData>, Edge> | null
  >(null);
  const lastProcessedFitTickRef = useRef(0);
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

  const handleDragOver = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
    if (
      !event.dataTransfer.types.includes(RECIPE_BLOCK_DND_MIME) &&
      !event.dataTransfer.types.includes("text/plain")
    ) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const handleDrop = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!reactFlowInstance) {
        return;
      }
      const raw =
        event.dataTransfer.getData(RECIPE_BLOCK_DND_MIME) ||
        event.dataTransfer.getData("text/plain");
      if (!raw) {
        return;
      }
      const payload = parseRecipeBlockDragPayload(raw);
      if (!payload) {
        return;
      }
      event.preventDefault();
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      if (payload.kind === "sampler") {
        addSamplerNode(payload.type as SamplerType, position, false);
        return;
      }
      if (payload.kind === "seed") {
        addSeedNode(payload.type as SeedBlockType, position, false);
        return;
      }
      if (payload.kind === "expression") {
        addExpressionNode(position, false);
        return;
      }
      if (payload.kind === "note") {
        addMarkdownNoteNode(position, false);
        return;
      }
      if (payload.type === "model_provider") {
        addModelProviderNode(position, false);
        return;
      }
      if (payload.type === "model_config") {
        addModelConfigNode(position, false);
        return;
      }
      addLlmNode(payload.type as LlmType, position, false);
    },
    [
      addExpressionNode,
      addLlmNode,
      addMarkdownNoteNode,
      addModelConfigNode,
      addModelProviderNode,
      addSamplerNode,
      addSeedNode,
      reactFlowInstance,
    ],
  );
  const getViewportCenterPosition = useCallback(() => {
    if (!reactFlowInstance || !flowContainerRef.current) {
      return undefined;
    }
    const rect = flowContainerRef.current.getBoundingClientRect();
    return reactFlowInstance.screenToFlowPosition({
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    });
  }, [reactFlowInstance]);
  const handleAddSamplerFromSheet = useCallback(
    (type: SamplerType) => {
      addSamplerNode(type, getViewportCenterPosition());
    },
    [addSamplerNode, getViewportCenterPosition],
  );
  const handleAddSeedFromSheet = useCallback(
    (type: SeedBlockType) => {
      addSeedNode(type, getViewportCenterPosition());
    },
    [addSeedNode, getViewportCenterPosition],
  );
  const handleAddLlmFromSheet = useCallback(
    (type: LlmType) => {
      addLlmNode(type, getViewportCenterPosition());
    },
    [addLlmNode, getViewportCenterPosition],
  );
  const handleAddModelProviderFromSheet = useCallback(() => {
    addModelProviderNode(getViewportCenterPosition());
  }, [addModelProviderNode, getViewportCenterPosition]);
  const handleAddModelConfigFromSheet = useCallback(() => {
    addModelConfigNode(getViewportCenterPosition());
  }, [addModelConfigNode, getViewportCenterPosition]);
  const handleAddExpressionFromSheet = useCallback(() => {
    addExpressionNode(getViewportCenterPosition());
  }, [addExpressionNode, getViewportCenterPosition]);
  const handleAddMarkdownNoteFromSheet = useCallback(() => {
    addMarkdownNoteNode(getViewportCenterPosition());
  }, [addMarkdownNoteNode, getViewportCenterPosition]);

  const configList = useMemo(() => Object.values(configs), [configs]);
  const config = activeConfigId ? configs[activeConfigId] : null;
  const dialogOptions = useMemo(
    () => buildDialogOptions(configList),
    [configList],
  );

  const handleToggleDirection = useCallback(() => {
    setLayoutDirection(layoutDirection === "LR" ? "TB" : "LR");
  }, [layoutDirection, setLayoutDirection]);

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
  const activeExecution = useMemo(
    () => pickLatestActiveExecution(executions),
    [executions],
  );
  const runtimeVisualState = useMemo(
    () =>
      deriveGraphRuntimeVisualState({
        activeExecution,
        configs,
        edges,
      }),
    [activeExecution, configs, edges],
  );
  const displayGraph = useMemo(
    () =>
      deriveDisplayGraph({
        nodes,
        edges,
        configs,
        layoutDirection,
        auxNodePositions,
        llmAuxVisibility,
        runtime: runtimeVisualState,
      }),
    [
      auxNodePositions,
      configs,
      edges,
      layoutDirection,
      llmAuxVisibility,
      nodes,
      runtimeVisualState,
    ],
  );
  const executionLocked = runtimeVisualState.executionLocked;
  const canvasInteractive = interactive && !executionLocked;
  const currentColumnConfig = useMemo(() => {
    const columnName = activeExecution?.current_column?.trim();
    if (!columnName) {
      return null;
    }
    for (const config of Object.values(configs)) {
      if (config.name.trim() === columnName) {
        return config;
      }
    }
    return null;
  }, [activeExecution?.current_column, configs]);
  const currentColumnIcon = useMemo(
    () => resolveExecutionColumnIcon(currentColumnConfig),
    [currentColumnConfig],
  );
  const displayNodeIds = useMemo(
    () => displayGraph.nodes.map((node) => node.id),
    [displayGraph.nodes],
  );

  const toggleInteractive = useCallback(() => {
    if (executionLocked) {
      return;
    }
    setInteractive((value) => !value);
  }, [executionLocked]);

  useEffect(() => {
    setExecutionLocked(executionLocked);
  }, [executionLocked, setExecutionLocked]);

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
    if (!reactFlowInstance || fitViewTick === 0 || activeView !== "editor") {
      return;
    }
    if (lastProcessedFitTickRef.current === fitViewTick) {
      return;
    }
    lastProcessedFitTickRef.current = fitViewTick;
    let frame2 = 0;
    let frame3 = 0;
    const frame1 = window.requestAnimationFrame(() => {
      frame2 = window.requestAnimationFrame(() => {
        frame3 = window.requestAnimationFrame(() => {
          reactFlowInstance.fitView({
            duration: 320,
            nodes: getFitNodeIdsIgnoringNotes(reactFlowInstance.getNodes()),
          });
        });
      });
    });
    return () => {
      window.cancelAnimationFrame(frame1);
      if (frame2) {
        window.cancelAnimationFrame(frame2);
      }
      if (frame3) {
        window.cancelAnimationFrame(frame3);
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
          <div className="h-[75vh] w-full rounded-t-none" ref={flowContainerRef}>
            {activeView === "editor" ? (
              <ReactFlow<Node<RecipeNodeData | RecipeGraphAuxNodeData>, Edge>
                onInit={setReactFlowInstance}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
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
                nodesDraggable={canvasInteractive}
                nodesConnectable={canvasInteractive}
                elementsSelectable={canvasInteractive}
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
                    onAddSampler={handleAddSamplerFromSheet}
                    onAddSeed={handleAddSeedFromSheet}
                    onAddLlm={handleAddLlmFromSheet}
                    onAddModelProvider={handleAddModelProviderFromSheet}
                    onAddModelConfig={handleAddModelConfigFromSheet}
                    onAddExpression={handleAddExpressionFromSheet}
                    onAddMarkdownNote={handleAddMarkdownNoteFromSheet}
                    onOpenProcessors={openProcessorsFromSheet}
                    copied={copied}
                    onCopy={copyRecipe}
                    onImport={() => setImportOpen(true)}
                  />
                </Panel>
                <ViewportControls
                  interactive={canvasInteractive}
                  lockDisabled={executionLocked}
                  onToggleInteractive={toggleInteractive}
                />
                {runtimeVisualState.batch && (
                  <Panel position="top-center" className="m-3">
                    <div className="rounded-lg border border-border/70 bg-card/95 px-3 py-2 text-xs shadow-sm">
                      <p className="font-medium text-foreground">
                        Batch {runtimeVisualState.batch.idx ?? "--"}/
                        {runtimeVisualState.batch.total}
                      </p>
                      {activeExecution?.current_column && (
                        <div className="mt-0.5 flex items-center gap-1.5 text-muted-foreground">
                          <HugeiconsIcon icon={currentColumnIcon} className="size-3.5" />
                          <p>Column: {activeExecution.current_column}</p>
                        </div>
                      )}
                    </div>
                  </Panel>
                )}
                <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center">
                  <div className="pointer-events-auto flex items-center gap-2">
                    <Button
                      type="button"
                      className="h-11 px-5"
                      onClick={() => openRunDialog(runDialogKind)}
                      disabled={previewLoading || fullLoading || executionLocked}
                    >
                      <HugeiconsIcon icon={CookBookIcon} className="size-4" />
                      {previewLoading || fullLoading || executionLocked
                        ? "Running..."
                        : "Run"}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      className="h-11 px-5"
                      onClick={() => {
                        openRunDialog(runDialogKind);
                        void validateFromDialog();
                      }}
                      disabled={validateLoading || executionLocked}
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
        readOnly={executionLocked}
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
