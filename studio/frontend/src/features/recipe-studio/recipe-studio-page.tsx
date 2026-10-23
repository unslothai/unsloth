// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DocumentAttachmentIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Background,
  BackgroundVariant,
  type Edge,
  getNodesBounds,
  type EdgeTypes,
  type Node,
  type NodeChange,
  type NodeTypes,
  Panel,
  ReactFlow,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import "@xyflow/react/dist/style.css";
import { Button } from "@/components/ui/button";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { buildRecipeEditorTourSteps } from "./tour";
import { BlockSheet } from "./components/block-sheet";
import { CANVAS_DELETE_KEYS } from "./constants";
import { LayoutControls } from "./components/controls/layout-controls";
import {
  SuperCardBreadcrumbBar,
  SuperCardCombineToolbar,
} from "./components/controls/super-card-controls";
import { RunValidateFloatingControls } from "./components/controls/run-validate-floating-controls";
import { ViewportControls } from "./components/controls/viewport-controls";
import { ExecutionsView } from "./components/executions/executions-view";
import { InternalsSync } from "./components/graph/internals-sync";
import {
  RecipeGraphAuxNode,
  type RecipeGraphAuxNodeData,
} from "./components/recipe-graph-aux-node";
import { RecipeNode } from "./components/recipe-graph-node";
import { RecipeGraphSuperCardNode } from "./components/recipe-graph-super-card-node";
import { RecipeGraphSemanticEdge } from "./components/recipe-graph-semantic-edge";
import { RecipeStudioHeader } from "./components/recipe-studio-header";
import { DataEdge } from "./components/rf-ui/data-edge";
import { ExecutionProgressIsland } from "./components/runtime/execution-progress-island";
import { ConfigDialog } from "./dialogs/config-dialog";
import { ImportDialog } from "./dialogs/import-dialog";
import { RunDialog } from "./dialogs/preview-dialog";
import { ProcessorsDialog } from "./dialogs/processors-dialog";
import { GithubCrawlerEasyView } from "./easy/github-crawler-easy-view";
import type {
  RecipeExecutionRecord,
  RecipeStudioView,
} from "./execution-types";
import { isExecutionInProgress } from "./executions/execution-helpers";
import { useRecipeEditorGraph } from "./hooks/use-recipe-editor-graph";
import { useRecipeRuntimeVisuals } from "./hooks/use-recipe-runtime-visuals";
import { useRecipeStudioActions } from "./hooks/use-recipe-studio-actions";
import { useRecipeStudioStore } from "./stores/recipe-studio";
import type { RecipeNodeData } from "./types";
import { getGraphWarnings } from "./utils/graph-warnings";
import {
  FIT_VIEW_DURATION_MS,
  FIT_VIEW_MAX_ZOOM,
  FIT_VIEW_PADDING,
  getFitViewTargetNodes,
} from "./utils/graph/fit-view";
import {
  SUPER_CARD_NODE_TYPE,
  applySuperCardView,
  getCombinableSelection,
  superCardFocusIds,
} from "./utils/graph/super-cards";
import { buildRecipePayload } from "./utils/payload";
import type { RecipePayload } from "./utils/payload/types";
import { buildDefaultSchemaTransform } from "./utils/processors";
import { buildDialogOptions } from "./utils/recipe-studio-view";

const SUPER_CARD_FOCUS_MIN_ZOOM = 0.8;

const NODE_TYPES: NodeTypes = {
  builder: RecipeNode,
  aux: RecipeGraphAuxNode,
  [SUPER_CARD_NODE_TYPE]: RecipeGraphSuperCardNode,
};
const EDGE_TYPES: EdgeTypes = {
  canvas: DataEdge,
  semantic: RecipeGraphSemanticEdge,
};
const COMPLETE_ISLAND_VISIBLE_MS = 7_000;
const TAB_SWITCH_FIT_DELAY_MS = 110;
/**
 * Max RAF iterations to wait for React Flow's ResizeObserver to populate
 * `node.measured` before calling fitView. ~20 frames ≈ 333 ms at 60 fps,
 * ample for the render → layout → ResizeObserver cycle.
 */
const MAX_FIT_VIEW_RETRIES = 20;
/**
 * Extra stable frames to wait after target nodes appear measured before
 * firing fitView, absorbing `updateNodeInternals` calls from InternalsSync
 * and node mount effects that can transiently reset measurements.
 */
const FIT_VIEW_STABLE_FRAMES = 3;

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
    sheetOpen,
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
    addToolProfileNode,
    addExpressionNode,
    addValidatorNode,
    addMarkdownNoteNode,
    selectConfig,
    openConfig,
    updateConfig,
    isValidConnection,
    setSheetOpen,
    setSheetView,
    setProcessors,
    setDialogOpen,
    resetRecipe,
    loadRecipe,
    setLayoutDirection,
    applyLayout,
    setAuxNodePosition,
    setExecutionLocked,
    superCards,
    openSuperCardId,
    selectedSuperCardIds,
    combineSelectionIntoSuperCard,
    ungroupSuperCard,
    openSuperCard,
    renameSuperCard,
    setSuperCardPosition,
    setSelectedSuperCardIds,
  } = useRecipeStudioStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      auxNodePositions: state.auxNodePositions,
      llmAuxVisibility: state.llmAuxVisibility,
      configs: state.configs,
      processors: state.processors,
      sheetOpen: state.sheetOpen,
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
      addToolProfileNode: state.addToolProfileNode,
      addExpressionNode: state.addExpressionNode,
      addValidatorNode: state.addValidatorNode,
      addMarkdownNoteNode: state.addMarkdownNoteNode,
      selectConfig: state.selectConfig,
      openConfig: state.openConfig,
      updateConfig: state.updateConfig,
      isValidConnection: state.isValidConnection,
      setSheetOpen: state.setSheetOpen,
      setSheetView: state.setSheetView,
      setProcessors: state.setProcessors,
      setDialogOpen: state.setDialogOpen,
      resetRecipe: state.resetRecipe,
      loadRecipe: state.loadRecipe,
      setLayoutDirection: state.setLayoutDirection,
      applyLayout: state.applyLayout,
      setAuxNodePosition: state.setAuxNodePosition,
      setExecutionLocked: state.setExecutionLocked,
      superCards: state.superCards,
      openSuperCardId: state.openSuperCardId,
      selectedSuperCardIds: state.selectedSuperCardIds,
      combineSelectionIntoSuperCard: state.combineSelectionIntoSuperCard,
      ungroupSuperCard: state.ungroupSuperCard,
      openSuperCard: state.openSuperCard,
      renameSuperCard: state.renameSuperCard,
      setSuperCardPosition: state.setSuperCardPosition,
      setSelectedSuperCardIds: state.setSelectedSuperCardIds,
    })),
  );
  const [sheetContainer, setSheetContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const flowContainerRef = useRef<HTMLDivElement | null>(null);
  const supportsEasyMode =
    initialPayload?.ui?.seed_source_type === "github_repo" ||
    (initialPayload?.recipe?.seed_config as { source?: { seed_type?: string } } | undefined)
      ?.source?.seed_type === "github_repo";
  const viewModeStorageKey = `recipe-studio:view-mode:${recipeId}`;
  const [activeView, setActiveViewState] = useState<RecipeStudioView>(() => {
    if (typeof window !== "undefined") {
      const stored = window.localStorage.getItem(viewModeStorageKey);
      if (stored === "easy" && supportsEasyMode) return "easy";
      if (stored === "editor" || stored === "executions") return stored;
    }
    return supportsEasyMode ? "easy" : "editor";
  });
  const setActiveView = useCallback(
    (next: RecipeStudioView | ((prev: RecipeStudioView) => RecipeStudioView)) => {
      setActiveViewState((prev) => {
        const resolved = typeof next === "function" ? next(prev) : next;
        if (typeof window !== "undefined") {
          window.localStorage.setItem(viewModeStorageKey, resolved);
        }
        return resolved;
      });
    },
    [viewModeStorageKey],
  );
  // Easy mode has no canvas overlay/progress island, so a started run would leave the Run button
  // stuck on "Running..." with nothing else changing. Flip to the Runs pane where progress is
  // rendered. Advanced (editor) keeps its island and stays put.
  const handleExecutionStart = useCallback(() => {
    setActiveView((currentView) =>
      currentView === "easy" ? "executions" : currentView,
    );
  }, [setActiveView]);
  const [processorsOpen, setProcessorsOpen] = useState(false);
  const [interactive, setInteractive] = useState(true);
  const [maximized, setMaximized] = useState(false);
  const [runtimeIslandMinimized, setRuntimeIslandMinimized] = useState(false);
  const [recentCompletedExecution, setRecentCompletedExecution] =
    useState<RecipeExecutionRecord | null>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance<
    Node<RecipeNodeData | RecipeGraphAuxNodeData>,
    Edge
  > | null>(null);
  const lastProcessedFitTickRef = useRef(0);
  const previousActiveViewRef = useRef<RecipeStudioView>("editor");
  const previousActiveExecutionIdRef = useRef<string | null>(null);
  const pendingEditorTabFitRef = useRef(false);
  const forceEditorTabFitRef = useRef(false);
  const viewportMovedSinceAutoFitRef = useRef(true);
  const {
    handleNodeClick,
    handleNodeDoubleClick,
    handleNodesChange,
    handleEdgesChange,
    handleDragOver,
    handleDrop,
    handleAddSamplerFromSheet,
    handleAddSeedFromSheet,
    handleAddLlmFromSheet,
    handleAddModelProviderFromSheet,
    handleAddModelConfigFromSheet,
    handleAddToolProfileFromSheet,
    handleAddExpressionFromSheet,
    handleAddValidatorFromSheet,
    handleAddMarkdownNoteFromSheet,
  } = useRecipeEditorGraph({
    nodes,
    edges,
    configs,
    reactFlowInstance,
    flowContainerRef,
    selectConfig,
    openConfig,
    onNodesChange,
    onEdgesChange,
    setAuxNodePosition,
    addSamplerNode,
    addSeedNode,
    addLlmNode,
    addModelProviderNode,
    addModelConfigNode,
    addToolProfileNode,
    addExpressionNode,
    addValidatorNode,
    addMarkdownNoteNode,
  });

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
        superCards,
      ),
    [
      auxNodePositions,
      configs,
      edges,
      layoutDirection,
      nodes,
      processors,
      superCards,
    ],
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
      state.superCards,
    ).payload;
  }, []);
  const {
    initialRecipeReady,
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
    fullRunName,
    setPreviewRows,
    setFullRows,
    setFullRunName,
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
    runPreview,
    runFull,
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
  });
  const {
    activeExecution,
    runtimeVisualState,
    displayGraph: flatDisplayGraph,
    currentColumnIcon,
  } = useRecipeRuntimeVisuals({
    executions,
    configs,
    nodes,
    edges,
    layoutDirection,
    auxNodePositions,
    llmAuxVisibility,
  });
  // Cards live outside the store, so keep React Flow's measurements here: a
  // card handed back without them is hidden for a frame and eats clicks.
  const superCardSizesRef = useRef(
    new Map<string, { width: number; height: number }>(),
  );
  const displayGraph = useMemo(() => {
    const view = applySuperCardView({
      graph: flatDisplayGraph,
      superCards,
      openSuperCardId,
      selectedSuperCardIds: new Set(selectedSuperCardIds),
      configs,
      edges,
      layoutDirection,
      executions,
      runtime: runtimeVisualState,
    });
    return {
      ...view,
      nodes: view.nodes.map((node) => {
        const measured =
          node.type === SUPER_CARD_NODE_TYPE
            ? superCardSizesRef.current.get(node.id)
            : undefined;
        return measured ? { ...node, measured } : node;
      }),
    };
  }, [
    configs,
    edges,
    executions,
    flatDisplayGraph,
    layoutDirection,
    openSuperCardId,
    runtimeVisualState,
    selectedSuperCardIds,
    superCards,
  ]);
  const displayNodeIds = useMemo(
    () => displayGraph.nodes.map((node) => node.id),
    [displayGraph.nodes],
  );
  const combinableIds = useMemo(
    () => (openSuperCardId ? [] : getCombinableSelection(nodes, superCards)),
    [nodes, openSuperCardId, superCards],
  );
  const handleCanvasNodesChange = useCallback(
    (changes: NodeChange<Node<RecipeNodeData | RecipeGraphAuxNodeData>>[]) => {
      const rest: typeof changes = [];
      let selection: string[] | null = null;
      for (const change of changes) {
        if (!("id" in change && superCards[change.id])) {
          rest.push(change);
          continue;
        }
        if (change.type === "dimensions" && change.dimensions) {
          superCardSizesRef.current.set(change.id, change.dimensions);
        } else if (change.type === "position" && change.position) {
          setSuperCardPosition(change.id, change.position);
        } else if (change.type === "select") {
          selection ??= [...selectedSuperCardIds];
          selection = change.selected
            ? [...new Set([...selection, change.id])]
            : selection.filter((id) => id !== change.id);
        }
      }
      if (selection) {
        setSelectedSuperCardIds(selection);
      }
      if (rest.length > 0) {
        handleNodesChange(rest);
      }
    },
    [
      handleNodesChange,
      selectedSuperCardIds,
      setSelectedSuperCardIds,
      setSuperCardPosition,
      superCards,
    ],
  );
  const handleCanvasNodeDoubleClick = useCallback(
    (
      event: Parameters<typeof handleNodeDoubleClick>[0],
      node: Node<RecipeNodeData | RecipeGraphAuxNodeData>,
    ) => {
      if (node.type === SUPER_CARD_NODE_TYPE) {
        openSuperCard(node.id);
        return;
      }
      handleNodeDoubleClick(event, node);
    },
    [handleNodeDoubleClick, openSuperCard],
  );
  const tourSteps = useMemo(
    () =>
      buildRecipeEditorTourSteps({
        // A placeholder stands in until the recipe loads, so the canvas steps would hit nothing.
        isGraphView: activeView === "editor" && initialRecipeReady,
        supportsEasyMode,
      }),
    [activeView, initialRecipeReady, supportsEasyMode],
  );
  const tour = useGuidedTourController({
    id: "recipe-editor",
    steps: tourSteps,
  });

  const executionLocked = runtimeVisualState.executionLocked;
  const canvasInteractive = interactive && !executionLocked;
  const runBusy = previewLoading || fullLoading || executionLocked;
  const islandExecution = activeExecution ?? recentCompletedExecution;

  // Easy mode uses runFull (artifact persisted, tracked in Runs pane), which
  // requires a non-empty fullRunName. The Easy form has no run-name input, so
  // seed a default once Easy is active; user can rename from Advanced/Runs.
  useEffect(() => {
    if (!supportsEasyMode) return;
    if (activeView !== "easy") return;
    if (fullRunName.trim()) return;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 16);
    const base = workflowName.trim() || "Easy run";
    setFullRunName(`${base} ${stamp}`);
  }, [supportsEasyMode, activeView, fullRunName, workflowName, setFullRunName]);

  const toggleInteractive = useCallback(() => {
    if (executionLocked) {
      return;
    }
    setInteractive((value) => !value);
  }, [executionLocked]);

  useEffect(() => {
    setExecutionLocked(executionLocked);
  }, [executionLocked, setExecutionLocked]);

  useEffect(() => {
    const activeExecutionId = activeExecution?.id ?? null;
    if (
      activeExecutionId &&
      activeExecutionId !== previousActiveExecutionIdRef.current
    ) {
      setRuntimeIslandMinimized(false);
    }
    previousActiveExecutionIdRef.current = activeExecutionId;
  }, [activeExecution?.id]);

  useEffect(() => {
    if (activeExecution) {
      setRecentCompletedExecution(null);
      return;
    }
    const latestCompleted = executions.find(
      (execution) =>
        execution.status === "completed" &&
        typeof execution.finishedAt === "number",
    );
    if (!latestCompleted || typeof latestCompleted.finishedAt !== "number") {
      setRecentCompletedExecution(null);
      return;
    }
    const elapsedMs = Date.now() - latestCompleted.finishedAt;
    if (elapsedMs >= COMPLETE_ISLAND_VISIBLE_MS) {
      setRecentCompletedExecution(null);
      return;
    }
    setRecentCompletedExecution(latestCompleted);
    const hideTimer = window.setTimeout(() => {
      setRecentCompletedExecution(null);
      setActiveView((currentView) =>
        currentView === "editor" ? "executions" : currentView,
      );
    }, COMPLETE_ISLAND_VISIBLE_MS - elapsedMs);
    return () => {
      window.clearTimeout(hideTimer);
    };
  }, [activeExecution, executions]);

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
    setSheetOpen(true);
  }, [setSheetOpen, setSheetView]);
  const openSourceBlockSheet = useCallback(() => {
    setSheetView("seed");
    setSheetOpen(true);
  }, [setSheetOpen, setSheetView]);
  const runDialogRows = runDialogKind === "preview" ? previewRows : fullRows;
  const runDialogLoading =
    runDialogKind === "preview" ? previewLoading : fullLoading;

  const scheduleFitView = useCallback(
    ({
      delayMs = 0,
      nodeIds,
      minZoom,
    }: { delayMs?: number; nodeIds?: string[]; minZoom?: number } = {}) => {
      if (!reactFlowInstance) {
        // eslint-disable-next-line @typescript-eslint/no-empty-function
        return () => {
          /* no-op: instance not available */
        };
      }

      let timeoutId = 0;
      let frameId = 0;
      let cancelled = false;

      /** Check whether every primary workflow node has been measured. */
      const allTargetsMeasured = (targets: Node[]): boolean =>
        targets.length > 0 &&
        targets.every(
          (n) => n.measured?.width != null && n.measured?.height != null,
        );

      const getTargets = (): Node[] => {
        const nodes = reactFlowInstance.getNodes();
        if (!nodeIds) {
          return getFitViewTargetNodes(nodes);
        }
        const byId = new Map(nodes.map((node) => [node.id, node]));
        return nodeIds.flatMap((id) => {
          const node = byId.get(id);
          return node ? [node] : [];
        });
      };

      /** Execute fitView on the current primary workflow nodes. */
      const doFit = () => {
        const targets = getTargets();
        if (targets.length === 0) {
          return;
        }
        viewportMovedSinceAutoFitRef.current = false;
        // When the whole group won't fit at minZoom, centre the first target
        // at minZoom instead of letting a clamped fit push it off screen.
        const box = flowContainerRef.current?.getBoundingClientRect();
        const bounds = getNodesBounds(targets);
        const fitZoom = box
          ? Math.min(
              box.width / (bounds.width * (1 + 2 * FIT_VIEW_PADDING)),
              box.height / (bounds.height * (1 + 2 * FIT_VIEW_PADDING)),
            )
          : Number.POSITIVE_INFINITY;
        const centreFirst = minZoom !== undefined && fitZoom < minZoom;
        reactFlowInstance.fitView({
          duration: FIT_VIEW_DURATION_MS,
          maxZoom: centreFirst ? minZoom : FIT_VIEW_MAX_ZOOM,
          padding: FIT_VIEW_PADDING,
          nodes: (centreFirst ? targets.slice(0, 1) : targets).map((n) => ({
            id: n.id,
          })),
          ...(minZoom ? { minZoom } : {}),
        });
      };

      let retries = 0;
      let stableCount = 0;
      const poll = () => {
        if (cancelled) {
          return;
        }
        if (retries >= MAX_FIT_VIEW_RETRIES) {
          // Timed out: fit with whatever we have (graceful fallback).
          doFit();
          return;
        }
        const targets = getTargets();
        if (allTargetsMeasured(targets)) {
          stableCount++;
          // Extra frames after measurements appear let updateNodeInternals
          // (InternalsSync, node mount effects) settle.
          if (stableCount >= FIT_VIEW_STABLE_FRAMES) {
            doFit();
            return;
          }
        } else {
          // Measurements reset (e.g. by updateNodeInternals): restart counter.
          stableCount = 0;
        }
        retries++;
        frameId = window.requestAnimationFrame(poll);
      };

      const start = () => {
        frameId = window.requestAnimationFrame(poll);
      };

      if (delayMs > 0) {
        timeoutId = window.setTimeout(start, delayMs);
      } else {
        start();
      }

      return () => {
        cancelled = true;
        if (timeoutId) {
          window.clearTimeout(timeoutId);
        }
        if (frameId) {
          window.cancelAnimationFrame(frameId);
        }
      };
    },
    [reactFlowInstance],
  );

  // Frame the new card and what it wires to, but never below a readable zoom;
  // far neighbours can sit past the edge.
  const handleCombine = useCallback(() => {
    combineSelectionIntoSuperCard();
    const state = useRecipeStudioStore.getState();
    const cardId = state.selectedSuperCardIds[0];
    if (cardId && state.superCards[cardId]) {
      scheduleFitView({
        nodeIds: superCardFocusIds(cardId, state.superCards, state.edges),
        minZoom: SUPER_CARD_FOCUS_MIN_ZOOM,
      });
    }
  }, [combineSelectionIntoSuperCard, scheduleFitView]);

  const toggleMaximize = useCallback(() => {
    // The maximized surface is a fixed z-50 overlay that already covers the app sidebar
    // (z-10/z-20), so we don't touch the sidebar's own state — that state is persisted in pin mode
    // and mutating it here would leak the temporary collapse into the next page/session.
    setMaximized((prev) => !prev);
    // Container size changes; refit once the layout settles.
    scheduleFitView({ delayMs: TAB_SWITCH_FIT_DELAY_MS });
  }, [scheduleFitView]);

  useEffect(() => {
    if (
      previousActiveViewRef.current !== activeView &&
      activeView === "editor"
    ) {
      pendingEditorTabFitRef.current = true;
      forceEditorTabFitRef.current =
        previousActiveViewRef.current === "executions";
    }
    previousActiveViewRef.current = activeView;
  }, [activeView]);

  useEffect(() => {
    if (activeView !== "editor" && reactFlowInstance) {
      setReactFlowInstance(null);
    }
  }, [activeView, reactFlowInstance]);

  // The "Exit full view" control lives inside the editor canvas, which unmounts on other tabs. Drop
  // full-view mode when leaving the editor so Easy/Runs aren't left under the fixed overlay.
  useEffect(() => {
    if (activeView !== "editor" && maximized) {
      setMaximized(false);
    }
  }, [activeView, maximized]);

  useEffect(() => {
    if (
      !reactFlowInstance ||
      activeView !== "editor" ||
      !pendingEditorTabFitRef.current
    ) {
      return;
    }
    pendingEditorTabFitRef.current = false;
    const forceFit = forceEditorTabFitRef.current;
    forceEditorTabFitRef.current = false;
    if (!(forceFit || viewportMovedSinceAutoFitRef.current)) {
      return;
    }
    return scheduleFitView({ delayMs: TAB_SWITCH_FIT_DELAY_MS });
  }, [activeView, reactFlowInstance, scheduleFitView]);

  useEffect(() => {
    if (!reactFlowInstance || fitViewTick === 0 || activeView !== "editor") {
      return;
    }
    if (lastProcessedFitTickRef.current === fitViewTick) {
      return;
    }
    lastProcessedFitTickRef.current = fitViewTick;
    return scheduleFitView();
  }, [activeView, fitViewTick, reactFlowInstance, scheduleFitView]);

  let editorContent: ReactElement;
  if (initialRecipeReady) {
    const openCard = openSuperCardId ? superCards[openSuperCardId] : null;
    editorContent = (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {openCard && (
          <SuperCardBreadcrumbBar
            card={openCard}
            recipeName={workflowName}
            locked={executionLocked}
            onExit={() => openSuperCard(null)}
            onRename={renameSuperCard}
            onUngroup={ungroupSuperCard}
          />
        )}
        <div className="relative min-h-0 flex-1">
          <ReactFlow<Node<RecipeNodeData | RecipeGraphAuxNodeData>, Edge>
            onInit={setReactFlowInstance}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            nodes={displayGraph.nodes}
            edges={displayGraph.edges}
            proOptions={{ hideAttribution: true }}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            defaultEdgeOptions={{
              type: "canvas",
              data: { path: "smoothstep" },
            }}
            onNodesChange={handleCanvasNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={onConnect}
            onNodeClick={handleNodeClick}
            onNodeDoubleClick={handleCanvasNodeDoubleClick}
            deleteKeyCode={CANVAS_DELETE_KEYS}
            isValidConnection={isValidConnection}
            onMoveEnd={(event) => {
              if (event) {
                viewportMovedSinceAutoFitRef.current = true;
              }
            }}
            nodesDraggable={canvasInteractive}
            nodesConnectable={canvasInteractive}
            elementsSelectable={canvasInteractive}
            fitView={false}
            className="@container/canvas h-full w-full rounded-t-none"
          >
            <LayoutControls
              direction={layoutDirection}
              onLayout={applyLayout}
              onToggleDirection={handleToggleDirection}
            />
            <SuperCardCombineToolbar
              nodeIds={combinableIds}
              locked={executionLocked}
              onCombine={handleCombine}
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
                <div className="pointer-events-auto w-full max-w-md rounded-2xl border border-dashed border-border/70 bg-background/80 px-6 py-6 text-center shadow-border backdrop-blur-[1px]">
                  <div className="mx-auto flex size-12 items-center justify-center corner-squircle rounded-md border border-border/70 bg-muted/40">
                    <HugeiconsIcon
                      icon={DocumentAttachmentIcon}
                      className="size-6 text-muted-foreground"
                    />
                  </div>
                  <div className="mt-4 space-y-2">
                    <p className="text-ui-11 font-semibold uppercase tracking-wide text-primary">
                      Best place to start
                    </p>
                    <p className="text-sm font-semibold text-foreground">
                      Start with source data
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Most synthetic-data recipes begin with a document,
                      dataset, or file before adding generation and checks.
                    </p>
                  </div>
                  <div className="mt-5 flex flex-col justify-center gap-2 sm:flex-row">
                    <Button
                      type="button"
                      className="corner-squircle"
                      onClick={openSourceBlockSheet}
                    >
                      <HugeiconsIcon
                        icon={DocumentAttachmentIcon}
                        className="size-4"
                      />
                      Start with source data
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      className="corner-squircle"
                      onClick={openRootBlockSheet}
                    >
                      <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                      Browse all steps
                    </Button>
                  </div>
                </div>
              </div>
            )}
            <Panel
              position="top-right"
              className="m-3"
              data-tour="recipe-add-step"
            >
              <BlockSheet
                container={sheetContainer}
                sheetView={sheetView}
                onViewChange={setSheetView}
                open={sheetOpen}
                onOpenChange={setSheetOpen}
                onAddSampler={handleAddSamplerFromSheet}
                onAddSeed={handleAddSeedFromSheet}
                onAddLlm={handleAddLlmFromSheet}
                onAddModelProvider={handleAddModelProviderFromSheet}
                onAddModelConfig={handleAddModelConfigFromSheet}
                onAddToolProfile={handleAddToolProfileFromSheet}
                onAddExpression={handleAddExpressionFromSheet}
                onAddValidator={handleAddValidatorFromSheet}
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
              maximized={maximized}
              onToggleMaximize={toggleMaximize}
            />
            {islandExecution &&
              (isExecutionInProgress(islandExecution.status) ||
                islandExecution.status === "completed") && (
                <Panel position="top-center" className="!m-0">
                  <ExecutionProgressIsland
                    execution={islandExecution}
                    currentColumnIcon={currentColumnIcon}
                    minimized={runtimeIslandMinimized}
                    onMinimizedChange={setRuntimeIslandMinimized}
                    onViewExecutions={() => setActiveView("executions")}
                  />
                </Panel>
              )}
            <RunValidateFloatingControls
              runBusy={runBusy}
              runDialogKind={runDialogKind}
              validateLoading={validateLoading}
              executionLocked={executionLocked}
              onOpenRunDialog={openRunDialog}
              onValidate={() => {
                openRunDialog(runDialogKind);
                void validateFromDialog();
              }}
            />
          </ReactFlow>
        </div>
      </div>
    );
  } else {
    editorContent = (
      <div className="flex h-full items-center justify-center px-6">
        <div className="rounded-2xl border border-border/70 bg-background/80 px-5 py-4 text-center shadow-border backdrop-blur-[1px]">
          <p className="text-sm font-medium text-foreground">Loading recipe</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Restoring the studio graph and saved settings.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={
        maximized
          ? "fixed inset-x-0 bottom-0 z-50 flex flex-col bg-background"
          : "flex h-full min-h-0 flex-1 flex-col bg-background"
      }
      style={
        maximized
          ? {
              // Start below the custom/mac window titlebar so the header and
              // its controls aren't hidden under (or click-blocked by) it.
              top: "var(--studio-non-chat-content-top-inset, var(--studio-content-top-inset, 0px))",
            }
          : undefined
      }
    >
      <main className="flex min-h-0 w-full flex-1 flex-col">
        <GuidedTour {...tour.tourProps} />
        <div
          className="relative flex min-h-0 w-full flex-1 flex-col overflow-hidden border"
          ref={setSheetContainer}
        >
          <RecipeStudioHeader
            activeView={activeView}
            saveLoading={saveLoading}
            saveTone={saveTone}
            savedAtLabel={savedAtLabel}
            workflowName={workflowName}
            warnings={getGraphWarnings(configs, edges)}
            supportsEasyMode={supportsEasyMode}
            onWorkflowNameChange={setWorkflowName}
            onViewChange={setActiveView}
            onSaveRecipe={() => {
              void persistRecipe();
            }}
          />
          <div
            data-tour="recipe-canvas"
            className="flex min-h-0 w-full flex-1 rounded-t-none"
            ref={flowContainerRef}
          >
            {activeView === "easy" ? (
              <GithubCrawlerEasyView
                configs={configs}
                rows={fullRows}
                setRows={setFullRows}
                updateConfig={updateConfig}
                onRun={() => {
                  // Easy mode is a full run (artifact persisted, tracked in Runs) capped at the
                  // user's row count. runFull requires a non-empty fullRunName; the effect above
                  // populates one on mount so runFull's closure is current by click time.
                  void runFull();
                }}
                runLoading={fullLoading || executionLocked}
                runErrors={runErrors}
                onSwitchToAdvanced={() => setActiveView("editor")}
              />
            ) : activeView === "editor" ? (
              editorContent
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
        localProviderNames={dialogOptions.localProviderNames}
        toolProfileAliases={dialogOptions.toolProfileAliases}
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
        fullRunName={fullRunName}
        onFullRunNameChange={setFullRunName}
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
