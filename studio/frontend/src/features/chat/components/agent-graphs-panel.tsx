// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { useBlocker } from "@tanstack/react-router";
import {
  CheckCircle2,
  Loader2,
  Pause,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Square,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type AgentGraphApproval,
  type AgentGraphDocument,
  type AgentGraphEvent,
  type AgentGraphNodeExecution,
  type AgentGraphRevision,
  type AgentGraphRevisionSummary,
  type AgentGraphRun,
  type AgentGraphSummary,
  cancelAgentGraphRun,
  createAgentGraph,
  decideAgentGraphApproval,
  deleteAgentGraph,
  getAgentGraph,
  getAgentGraphRun,
  listAgentGraphEvents,
  listAgentGraphRevisions,
  listAgentGraphRuns,
  listAgentGraphs,
  pauseAgentGraphRun,
  resumeAgentGraphRun,
  retryAgentGraphRun,
  startAgentGraphRun,
  startQueuedAgentGraphRun,
  updateAgentGraph,
  validateAgentGraph,
} from "../api/agent-workspace-api";
import { AgentGraphEditor } from "./agent-graph-editor";
import { AgentGraphDraftStatus, AgentGraphLiveStatus } from "./agent-graph-ui";
import {
  agentGraphRouteNavigationShouldBlock,
  graphDraftIsPending,
  graphEditorIsReady,
  graphLoadCanApply,
  graphMutationIsCurrent,
  graphResponseIsCurrent,
  graphRunActions,
  graphRunMatchesEditor,
  graphRunResponseIsCurrent,
  registerAgentGraphDraftNavigationGuard,
} from "./agent-graph-workflow-state";
import { safeAgentWorkspaceError } from "./agent-workspace-state";

const SAMPLE_GRAPH: AgentGraphDocument = {
  name: "New graph",
  description: "",
  inputSchema: { type: "object" },
  outputSchema: { type: "object" },
  nodes: [
    { id: "input", type: "input", config: { name: "input" } },
    {
      id: "loop",
      type: "loop",
      config: {
        instruction: "Work on {input}",
        runtime: {
          kind: "local",
          model: "",
          permissionMode: "off",
          maxOutputTokens: 8192,
        },
      },
    },
    { id: "output", type: "output", config: { name: "output" } },
  ],
  edges: [
    { from: "input", to: "loop" },
    { from: "loop", to: "output" },
  ],
  permissions: { allowedToolServerIds: [] },
  limits: {
    maxNodes: 100,
    maxRunSeconds: 3600,
    maxOutputBytes: 1048576,
    maxIterations: 100,
    maxOutputTokens: 262144,
  },
};
const SAMPLE_RUN_INPUT = '{\n  "task": "inspect this project"\n}';

const TERMINAL_RUNS = new Set([
  "cancelled",
  "completed",
  "failed",
  "interrupted",
]);

function statusVariant(
  status: string,
): "secondary" | "outline" | "destructive" {
  if (["failed", "interrupted", "rejected"].includes(status)) {
    return "destructive";
  }
  if (["completed", "approved"].includes(status)) {
    return "secondary";
  }
  return "outline";
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function documentFromRevision(
  revision: AgentGraphRevision,
): AgentGraphDocument {
  return {
    name: revision.name,
    description: revision.description,
    metadata: revision.metadata,
    inputSchema: revision.inputSchema,
    outputSchema: revision.outputSchema,
    nodes: revision.nodes.map((node) => ({
      ...node,
      retryPolicy: node.retryPolicy ?? {
        maxAttempts: 1,
        backoffMs: 0,
        retryOn: ["error", "timeout"],
      },
    })),
    edges: revision.edges,
    permissions: revision.permissions,
    limits: {
      ...revision.limits,
      maxIterations: revision.limits.maxIterations ?? 100,
      maxOutputTokens: revision.limits.maxOutputTokens ?? 262144,
    },
  };
}

export function AgentGraphsPanel({ projectId }: { projectId: string }) {
  const [graphs, setGraphs] = useState<AgentGraphSummary[]>([]);
  const [selectedGraph, setSelectedGraph] = useState<AgentGraphSummary | null>(
    null,
  );
  const [graphDocument, setGraphDocument] =
    useState<AgentGraphDocument>(SAMPLE_GRAPH);
  const [documentText, setDocumentText] = useState(() => pretty(SAMPLE_GRAPH));
  const [dirty, setDirty] = useState(false);
  const [editorHasDrafts, setEditorHasDrafts] = useState(false);
  const [jsonDraftDirty, setJsonDraftDirty] = useState(false);
  const [editorRevision, setEditorRevision] = useState(0);
  const [editorEpoch, setEditorEpoch] = useState(0);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [loadedGraphId, setLoadedGraphId] = useState<string | null>(null);
  const [validation, setValidation] = useState<string | null>(null);
  const [revisions, setRevisions] = useState<AgentGraphRevisionSummary[]>([]);
  const [inputText, setInputText] = useState(SAMPLE_RUN_INPUT);
  const [inputDraftDirty, setInputDraftDirty] = useState(false);
  const [runs, setRuns] = useState<AgentGraphRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<AgentGraphRun | null>(null);
  const [runNodes, setRunNodes] = useState<AgentGraphNodeExecution[]>([]);
  const [runEvents, setRunEvents] = useState<AgentGraphEvent[]>([]);
  const [approvals, setApprovals] = useState<AgentGraphApproval[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const graphListGeneration = useRef(0);
  const graphLoadGeneration = useRef(0);
  const runLoadGeneration = useRef(0);
  const mutationGeneration = useRef(0);
  const selectedGraphId = useRef<string | null>(null);
  const selectedGraphSummary = useRef<AgentGraphSummary | null>(null);
  const selectedRunId = useRef<string | null>(null);
  const hasUnsavedGraphDraft = graphDraftIsPending(
    dirty,
    editorHasDrafts,
    jsonDraftDirty,
  );
  const hasUnsavedDraft = graphDraftIsPending(
    dirty,
    editorHasDrafts,
    jsonDraftDirty,
    inputDraftDirty,
  );
  const hasUnsavedDraftRef = useRef(hasUnsavedDraft);
  const hasUnsavedGraphDraftRef = useRef(hasUnsavedGraphDraft);
  const editorReady = graphEditorIsReady(
    selectedGraph?.id ?? null,
    loadedGraphId,
  );
  const runActionState = graphRunActions(selectedRun?.status ?? "");
  const confirmDraftDiscard = useCallback(
    () =>
      !hasUnsavedDraftRef.current ||
      window.confirm("Discard the unsaved graph changes and editor drafts?"),
    [],
  );
  const shouldBlockRouteNavigation = useCallback(
    ({ next }: { next: { pathname: string; search: unknown } }) =>
      agentGraphRouteNavigationShouldBlock(
        next,
        projectId,
        confirmDraftDiscard,
      ),
    [confirmDraftDiscard, projectId],
  );

  useBlocker({
    shouldBlockFn: shouldBlockRouteNavigation,
    enableBeforeUnload: false,
  });

  useEffect(() => {
    hasUnsavedDraftRef.current = hasUnsavedDraft;
  }, [hasUnsavedDraft]);

  useEffect(() => {
    hasUnsavedGraphDraftRef.current = hasUnsavedGraphDraft;
  }, [hasUnsavedGraphDraft]);

  useEffect(() => {
    selectedGraphId.current = selectedGraph?.id ?? null;
    selectedGraphSummary.current = selectedGraph;
  }, [selectedGraph]);

  useEffect(() => {
    selectedRunId.current = selectedRun?.id ?? null;
  }, [selectedRun?.id]);

  useEffect(
    () => registerAgentGraphDraftNavigationGuard(confirmDraftDiscard),
    [confirmDraftDiscard],
  );

  useEffect(() => {
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!hasUnsavedDraftRef.current) {
        return;
      }
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, []);

  const updateEditorDocument = useCallback(
    (next: AgentGraphDocument, markDirty = true) => {
      setGraphDocument(next);
      setDocumentText(pretty(next));
      setDirty(markDirty);
      setJsonDraftDirty(false);
      setValidation(null);
    },
    [],
  );

  const loadGraphs = useCallback(async () => {
    const generation = ++graphListGeneration.current;
    setError(null);
    try {
      const next = await listAgentGraphs(projectId);
      if (generation !== graphListGeneration.current) {
        return;
      }
      setGraphs(next);
      const currentId = selectedGraphId.current;
      if (!currentId) {
        return;
      }
      const replacement = next.find((graph) => graph.id === currentId) ?? null;
      if (!replacement) {
        if (hasUnsavedDraftRef.current) {
          setError(
            "This graph was deleted elsewhere. Your unsaved draft is still open; copy it or select another graph when you are ready to discard it.",
          );
          return;
        }
        graphLoadGeneration.current += 1;
        selectedGraphId.current = null;
        selectedGraphSummary.current = null;
        setSelectedGraph(null);
        setLoadedGraphId(null);
        setRuns([]);
        setRevisions([]);
        runLoadGeneration.current += 1;
        setSelectedRun(null);
        setRunNodes([]);
        setRunEvents([]);
        setApprovals([]);
        setEditorHasDrafts(false);
        setJsonDraftDirty(false);
        setEditorRevision(0);
        setEditorEpoch((current) => current + 1);
        updateEditorDocument(SAMPLE_GRAPH, false);
        return;
      }
      if (hasUnsavedGraphDraftRef.current) {
        return;
      }
      const previous = selectedGraphSummary.current;
      if (
        previous?.id !== replacement.id ||
        previous.currentRevision !== replacement.currentRevision ||
        previous.updatedAt !== replacement.updatedAt
      ) {
        setLoadingGraph(true);
        setLoadedGraphId(null);
      }
      selectedGraphSummary.current = replacement;
      setSelectedGraph((current) => {
        if (
          current?.id === replacement.id &&
          current.currentRevision === replacement.currentRevision &&
          current.updatedAt === replacement.updatedAt
        ) {
          return current;
        }
        return replacement;
      });
    } catch (reason) {
      if (generation !== graphListGeneration.current) {
        return;
      }
      setError(safeAgentWorkspaceError(reason));
    }
  }, [projectId, updateEditorDocument]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadGraphs(), 0);
    return () => {
      window.clearTimeout(timer);
      graphListGeneration.current += 1;
      mutationGeneration.current += 1;
    };
  }, [loadGraphs]);

  useEffect(() => {
    const graphId = selectedGraph?.id;
    const graphRevision = selectedGraph?.currentRevision;
    const generation = ++graphLoadGeneration.current;
    if (!graphId || graphRevision === undefined) return;
    void getAgentGraph(projectId, graphId)
      .then((result) => {
        if (
          !graphLoadCanApply(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
            hasUnsavedGraphDraftRef.current,
          )
        ) {
          if (
            graphResponseIsCurrent(
              generation,
              graphLoadGeneration.current,
              graphId,
              selectedGraphId.current,
            ) &&
            hasUnsavedGraphDraftRef.current
          ) {
            setError(
              "The graph changed elsewhere while you were editing. Your draft was preserved; discard it before loading the remote revision.",
            );
            setLoadingGraph(false);
          }
          return;
        }
        updateEditorDocument(documentFromRevision(result.revision), false);
        setEditorRevision(result.revision.revision);
        setEditorEpoch((current) => current + 1);
        setEditorHasDrafts(false);
        setLoadedGraphId(graphId);
        setLoadingGraph(false);
      })
      .catch((reason) => {
        if (
          graphResponseIsCurrent(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
          )
        ) {
          setError(safeAgentWorkspaceError(reason));
          setLoadingGraph(false);
        }
      });
    void listAgentGraphRuns(projectId, graphId, 50)
      .then((nextRuns) => {
        if (
          graphResponseIsCurrent(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
          )
        ) {
          setRuns(nextRuns);
        }
      })
      .catch((reason) => {
        if (
          graphResponseIsCurrent(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
          )
        ) {
          setError(safeAgentWorkspaceError(reason));
        }
      });
    void listAgentGraphRevisions(projectId, graphId)
      .then((nextRevisions) => {
        if (
          graphResponseIsCurrent(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
          )
        ) {
          setRevisions(nextRevisions);
        }
      })
      .catch((reason) => {
        if (
          graphResponseIsCurrent(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
          )
        ) {
          setError(safeAgentWorkspaceError(reason));
        }
      });
    return () => {
      if (generation === graphLoadGeneration.current) {
        graphLoadGeneration.current += 1;
      }
    };
  }, [
    projectId,
    selectedGraph?.id,
    selectedGraph?.currentRevision,
    updateEditorDocument,
  ]);

  const refreshRun = useCallback(
    async (runId: string) => {
      const graphId = selectedGraph?.id;
      if (!graphId || runId !== selectedRunId.current) {
        return;
      }
      const generation = ++runLoadGeneration.current;
      setError(null);
      try {
        const [detail, events, nextRuns] = await Promise.all([
          getAgentGraphRun(projectId, runId),
          listAgentGraphEvents(projectId, runId),
          listAgentGraphRuns(projectId, graphId, 50),
        ]);
        if (
          !graphRunResponseIsCurrent(
            generation,
            runLoadGeneration.current,
            graphId,
            selectedGraphId.current,
            runId,
            selectedRunId.current,
          ) ||
          detail.run.graphId !== graphId
        ) {
          return;
        }
        setSelectedRun(detail.run);
        selectedRunId.current = detail.run.id;
        setRunNodes(detail.nodes);
        setApprovals(detail.approvals);
        setRunEvents(events);
        setRuns(nextRuns);
      } catch (reason) {
        if (
          graphRunResponseIsCurrent(
            generation,
            runLoadGeneration.current,
            graphId,
            selectedGraphId.current,
            runId,
            selectedRunId.current,
          )
        ) {
          setError(safeAgentWorkspaceError(reason));
        }
      }
    },
    [projectId, selectedGraph?.id],
  );

  useEffect(() => {
    if (!selectedRun || TERMINAL_RUNS.has(selectedRun.status)) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshRun(selectedRun.id);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [refreshRun, selectedRun]);

  const activeApproval = useMemo(
    () => approvals.find((approval) => approval.status === "pending"),
    [approvals],
  );

  const nodeStatuses = useMemo(() => {
    const statuses: Record<string, string> = {};
    if (
      !graphRunMatchesEditor(
        selectedRun,
        selectedGraph?.id ?? null,
        editorRevision,
      )
    ) {
      return statuses;
    }
    for (const execution of runNodes) {
      statuses[execution.nodeId] = execution.status;
    }
    return statuses;
  }, [editorRevision, runNodes, selectedGraph?.id, selectedRun]);

  async function action<T>(
    key: string,
    target: {
      projectId: string;
      graphId: string | null;
      runId?: string | null;
    },
    work: () => Promise<T>,
    complete?: (value: T) => Promise<void> | void,
  ) {
    if (busy) {
      return;
    }
    const generation = ++mutationGeneration.current;
    const isCurrent = () =>
      generation === mutationGeneration.current &&
      graphMutationIsCurrent(target, {
        projectId,
        graphId: selectedGraphId.current,
        runId: selectedRunId.current,
      });
    setBusy(key);
    setError(null);
    try {
      const result = await work();
      if (!isCurrent()) {
        return;
      }
      await complete?.(result);
      toast.success("Graph updated");
    } catch (reason) {
      if (!isCurrent()) {
        return;
      }
      const message = safeAgentWorkspaceError(reason);
      setError(message);
      toast.error("Graph action failed", { description: message });
    } finally {
      if (generation === mutationGeneration.current) {
        setBusy(null);
      }
    }
  }

  function clearRunSelection() {
    runLoadGeneration.current += 1;
    selectedRunId.current = null;
    setSelectedRun(null);
    setRunNodes([]);
    setRunEvents([]);
    setApprovals([]);
  }

  function selectRun(run: AgentGraphRun) {
    runLoadGeneration.current += 1;
    selectedRunId.current = run.id;
    setSelectedRun(run);
    setRunNodes([]);
    setRunEvents([]);
    setApprovals([]);
    void refreshRun(run.id);
  }

  function confirmDraftReplacement(): boolean {
    return (
      !hasUnsavedDraft ||
      window.confirm("Discard the unsaved graph changes and editor drafts?")
    );
  }

  function selectGraph(graph: AgentGraphSummary | null) {
    if (busy) {
      return;
    }
    if (graph?.id === selectedGraph?.id) {
      return;
    }
    if (!confirmDraftReplacement()) {
      return;
    }
    mutationGeneration.current += 1;
    setBusy(null);
    setError(null);
    graphLoadGeneration.current += 1;
    selectedGraphId.current = graph?.id ?? null;
    selectedGraphSummary.current = graph;
    setLoadingGraph(Boolean(graph));
    setLoadedGraphId(null);
    setSelectedGraph(graph);
    setRuns([]);
    setRevisions([]);
    clearRunSelection();
    setEditorHasDrafts(false);
    setJsonDraftDirty(false);
    setDirty(false);
    setInputText(SAMPLE_RUN_INPUT);
    setInputDraftDirty(false);
    setValidation(null);
    setEditorRevision(0);
    setEditorEpoch((current) => current + 1);
    if (!graph) {
      updateEditorDocument(SAMPLE_GRAPH, false);
    }
  }

  async function save() {
    if (!editorReady) {
      setError("Load this graph revision before saving it.");
      return;
    }
    if (editorHasDrafts || jsonDraftDirty) {
      setError(
        "Apply the visible node, schema, or JSON editor drafts before saving.",
      );
      return;
    }
    const graph = selectedGraph;
    const document = graphDocument;
    const target = { projectId, graphId: graph?.id ?? null };
    await action(
      "save",
      target,
      async () => {
        const checked = await validateAgentGraph(projectId, document);
        if (graph) {
          return updateAgentGraph(projectId, graph, checked.document);
        }
        return createAgentGraph(projectId, checked.document);
      },
      (savedGraph) => {
        setValidation("Valid graph contract");
        setDirty(false);
        setEditorHasDrafts(false);
        setJsonDraftDirty(false);
        setSelectedGraph(savedGraph);
        selectedGraphId.current = savedGraph.id;
        selectedGraphSummary.current = savedGraph;
        setLoadedGraphId(null);
        setLoadingGraph(true);
        void loadGraphs();
      },
    );
  }

  async function validateDraft() {
    if (!editorReady) {
      setError("Load this graph revision before validating it.");
      return;
    }
    if (editorHasDrafts || jsonDraftDirty) {
      setError(
        "Apply the visible node, schema, or JSON editor drafts before validating.",
      );
      return;
    }
    await action(
      "validate",
      { projectId, graphId: selectedGraph?.id ?? null },
      () => validateAgentGraph(projectId, graphDocument),
      () => {
        setValidation("Valid graph contract");
      },
    );
  }

  async function applyJsonDocument() {
    if (!editorReady) {
      setError("Load this graph revision before applying JSON.");
      return;
    }
    if (
      editorHasDrafts &&
      !window.confirm("Discard the unapplied node and schema editor drafts?")
    ) {
      return;
    }
    let parsed: AgentGraphDocument;
    try {
      parsed = JSON.parse(documentText) as AgentGraphDocument;
    } catch {
      setError("Graph document is not valid JSON.");
      return;
    }
    await action(
      "json",
      { projectId, graphId: selectedGraph?.id ?? null },
      () => validateAgentGraph(projectId, parsed),
      (result) => {
        setEditorHasDrafts(false);
        setEditorEpoch((current) => current + 1);
        updateEditorDocument(result.document);
        setValidation("Valid graph contract");
      },
    );
  }

  async function loadRevision(revision: number) {
    if (!selectedGraph) {
      return;
    }
    if (revision === editorRevision && !hasUnsavedDraft) {
      return;
    }
    if (!confirmDraftReplacement()) {
      return;
    }
    const graphId = selectedGraph.id;
    const generation = ++graphLoadGeneration.current;
    setLoadingGraph(true);
    clearRunSelection();
    await action(
      "revision",
      { projectId, graphId },
      async () => {
        try {
          return await getAgentGraph(projectId, graphId, revision);
        } catch (reason) {
          setLoadingGraph(false);
          throw reason;
        }
      },
      (result) => {
        if (
          !graphResponseIsCurrent(
            generation,
            graphLoadGeneration.current,
            graphId,
            selectedGraphId.current,
          )
        ) {
          return;
        }
        setEditorHasDrafts(false);
        setJsonDraftDirty(false);
        setEditorRevision(result.revision.revision);
        setEditorEpoch((current) => current + 1);
        setLoadedGraphId(graphId);
        setLoadingGraph(false);
        setInputText(SAMPLE_RUN_INPUT);
        setInputDraftDirty(false);
        updateEditorDocument(
          documentFromRevision(result.revision),
          revision !== selectedGraph.currentRevision,
        );
      },
    );
  }

  async function startRun() {
    if (!selectedGraph) {
      setError("Save a graph before starting a run.");
      return;
    }
    if (hasUnsavedGraphDraft) {
      setError("Save this revision before starting a test run.");
      return;
    }
    if (!editorReady) {
      setError("Load this graph revision before starting a test run.");
      return;
    }
    if (editorRevision < 1) {
      setError("Load a pinned graph revision before starting a test run.");
      return;
    }
    let input: Record<string, unknown>;
    try {
      input = JSON.parse(inputText) as Record<string, unknown>;
    } catch {
      setError("Run input is not valid JSON.");
      return;
    }
    const graphId = selectedGraph.id;
    await action(
      "run",
      { projectId, graphId },
      () =>
        startAgentGraphRun(projectId, graphId, {
          input,
          revision: editorRevision,
        }),
      (run) => {
        if (run.graphId !== graphId) {
          return;
        }
        selectedRunId.current = run.id;
        setSelectedRun(run);
        setInputDraftDirty(false);
        setRuns((current) => [
          run,
          ...current.filter((item) => item.id !== run.id),
        ]);
        void refreshRun(run.id);
      },
    );
  }

  async function refreshRunList() {
    const graphId = selectedGraph?.id;
    if (!graphId) return;
    const generation = ++runLoadGeneration.current;
    setError(null);
    try {
      const nextRuns = await listAgentGraphRuns(projectId, graphId, 50);
      if (
        graphResponseIsCurrent(
          generation,
          runLoadGeneration.current,
          graphId,
          selectedGraphId.current,
        )
      ) {
        setRuns(nextRuns);
      }
    } catch (reason) {
      if (
        graphResponseIsCurrent(
          generation,
          runLoadGeneration.current,
          graphId,
          selectedGraphId.current,
        )
      ) {
        setError(safeAgentWorkspaceError(reason));
      }
    }
  }

  async function removeGraph() {
    if (
      !(
        selectedGraph &&
        window.confirm(
          `Delete graph "${selectedGraph.name}" and its stopped run history?`,
        )
      )
    ) {
      return;
    }
    const graphId = selectedGraph.id;
    await action(
      "delete",
      { projectId, graphId },
      async () => {
        await deleteAgentGraph(projectId, graphId);
      },
      () => {
        setSelectedGraph(null);
        selectedGraphId.current = null;
        selectedGraphSummary.current = null;
        setLoadedGraphId(null);
        setLoadingGraph(false);
        setSelectedRun(null);
        setRuns([]);
        setRunNodes([]);
        setRunEvents([]);
        setApprovals([]);
        setRevisions([]);
        setEditorHasDrafts(false);
        setJsonDraftDirty(false);
        setInputText(SAMPLE_RUN_INPUT);
        setInputDraftDirty(false);
        setEditorRevision(0);
        setEditorEpoch((current) => current + 1);
        updateEditorDocument(SAMPLE_GRAPH, false);
        void loadGraphs();
      },
    );
  }

  async function runMutation(
    operation: (projectId: string, runId: string) => Promise<AgentGraphRun>,
  ) {
    if (!selectedRun) {
      return;
    }
    const runId = selectedRun.id;
    const graphId = selectedRun.graphId;
    await action(
      "run-mutation",
      { projectId, graphId, runId },
      () => operation(projectId, runId),
      (run) => {
        if (
          (run.id !== runId && run.retryOfRunId !== runId) ||
          run.graphId !== graphId
        ) {
          return;
        }
        selectedRunId.current = run.id;
        setSelectedRun(run);
        void refreshRun(run.id);
      },
    );
  }

  async function decide(
    approval: AgentGraphApproval,
    decision: "approved" | "rejected",
  ) {
    if (!selectedRun) {
      return;
    }
    const runId = selectedRun.id;
    const graphId = selectedRun.graphId;
    await action(
      "approval",
      { projectId, graphId, runId },
      () => decideAgentGraphApproval(projectId, runId, approval.id, decision),
      (decidedApproval) => {
        setApprovals((current) =>
          current.map((item) =>
            item.id === decidedApproval.id ? decidedApproval : item,
          ),
        );
        void refreshRun(runId);
      },
    );
  }

  return (
    <section className="rounded-[22px] border border-border/60 bg-card/35 px-4 py-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <Play className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-ui-14 font-semibold text-foreground">
            Sloth graphs
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Versioned project workflows built from the existing Loop runtime.
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Loop and model nodes need a durable runtime selection with a model
            before they can run.
          </p>
        </div>
        <Button
          type="button"
          size="xs"
          variant="ghost"
          onClick={() => void loadGraphs()}
          disabled={Boolean(busy)}
        >
          <RefreshCw className={busy === "refresh" ? "animate-spin" : ""} />{" "}
          Refresh
        </Button>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[250px_minmax(0,1fr)]">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium">Project graphs</p>
            <Button
              type="button"
              size="xs"
              variant="outline"
              onClick={() => selectGraph(null)}
              disabled={Boolean(busy)}
            >
              <Plus /> New
            </Button>
          </div>
          {graphs.map((graph) => (
            <button
              type="button"
              key={graph.id}
              onClick={() => selectGraph(graph)}
              disabled={Boolean(busy)}
              aria-pressed={selectedGraph?.id === graph.id}
              className={`w-full rounded-xl px-3 py-2 text-left text-xs ${selectedGraph?.id === graph.id ? "bg-muted" : "bg-muted/35"}`}
            >
              <span className="block truncate font-medium">{graph.name}</span>
              <span className="text-[11px] text-muted-foreground">
                Revision {graph.currentRevision}
              </span>
            </button>
          ))}
          {graphs.length === 0 ? (
            <p className="rounded-xl bg-muted/35 px-3 py-4 text-center text-xs text-muted-foreground">
              No graphs yet.
            </p>
          ) : null}
          {selectedGraph ? (
            <div className="rounded-xl border border-border/60 bg-background/45 p-3">
              <p className="text-xs font-medium">Revision history</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {revisions.map((revision) => (
                  <Button
                    type="button"
                    size="xs"
                    variant={
                      revision.revision === selectedGraph.currentRevision &&
                      !dirty
                        ? "secondary"
                        : "outline"
                    }
                    key={revision.revision}
                    onClick={() => void loadRevision(revision.revision)}
                    disabled={Boolean(busy) || loadingGraph}
                    aria-current={
                      revision.revision === editorRevision ? "true" : undefined
                    }
                    title={new Date(revision.createdAt).toLocaleString()}
                  >
                    r{revision.revision}
                  </Button>
                ))}
              </div>
            </div>
          ) : null}
          {selectedGraph ? (
            <div className="rounded-xl border border-border/60 bg-background/45 p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-medium">Runs</p>
                <Button
                  type="button"
                  size="icon-xs"
                  variant="ghost"
                  onClick={() => void refreshRunList()}
                  aria-label="Refresh graph runs"
                  disabled={Boolean(busy) || loadingGraph}
                >
                  <RefreshCw />
                </Button>
              </div>
              <div className="mt-2 space-y-1.5">
                {runs.slice(0, 12).map((run) => (
                  <button
                    type="button"
                    key={run.id}
                    onClick={() => selectRun(run)}
                    disabled={Boolean(busy)}
                    aria-pressed={selectedRun?.id === run.id}
                    className="flex w-full items-center gap-2 rounded-lg bg-muted/35 px-2.5 py-2 text-left text-[11px]"
                  >
                    <Badge variant={statusVariant(run.status)}>
                      {run.status}
                    </Badge>
                    <span className="min-w-0 flex-1 truncate">
                      {run.currentNodeId || run.id.slice(0, 8)}
                    </span>
                    <span className="text-muted-foreground">
                      r{run.revision}
                    </span>
                  </button>
                ))}
                {runs.length === 0 ? (
                  <p className="text-[11px] text-muted-foreground">
                    No runs yet.
                  </p>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-xs font-medium">Visual revision editor</p>
              <AgentGraphDraftStatus
                loading={loadingGraph}
                ready={editorReady}
                pending={hasUnsavedDraft}
                message={validation || "Pinned revision contract"}
              />
            </div>
            <div className="flex items-center gap-1.5">
              {selectedGraph ? (
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() => void removeGraph()}
                  disabled={Boolean(busy)}
                >
                  <Trash2 /> Delete
                </Button>
              ) : null}
              <Button
                type="button"
                size="xs"
                variant="outline"
                onClick={() => void validateDraft()}
                disabled={Boolean(busy) || loadingGraph || !editorReady}
              >
                {busy === "validate" ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <CheckCircle2 />
                )}{" "}
                Validate
              </Button>
              <Button
                type="button"
                size="xs"
                onClick={() => void save()}
                disabled={Boolean(busy) || loadingGraph || !editorReady}
              >
                {busy === "save" ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <CheckCircle2 />
                )}{" "}
                Save revision
              </Button>
            </div>
          </div>
          <AgentGraphEditor
            key={`${selectedGraph?.id ?? "new"}:${editorRevision}:${editorEpoch}`}
            document={graphDocument}
            onChange={updateEditorDocument}
            onDraftStateChange={setEditorHasDrafts}
            nodeStatuses={nodeStatuses}
            disabled={
              Boolean(busy) || loadingGraph || !editorReady || jsonDraftDirty
            }
          />
          {selectedGraph ? (
            <p className="text-[11px] text-muted-foreground">
              Saving creates revision {selectedGraph.currentRevision + 1}.
              Existing runs keep their pinned revision.
            </p>
          ) : (
            <p className="text-[11px] text-muted-foreground">
              The first save creates revision 1. The backend validates IDs,
              edges, cycles, reachability, node configs, permissions, and
              budgets.
            </p>
          )}
          <details className="rounded-xl border border-border/60 bg-background/45 p-3">
            <summary className="cursor-pointer text-[11px] font-medium">
              Advanced graph contract JSON
            </summary>
            <Textarea
              value={documentText}
              onChange={(event) => {
                setDocumentText(event.target.value);
                setJsonDraftDirty(true);
                setValidation(null);
              }}
              aria-label="Graph revision JSON"
              className="mt-2 min-h-[260px] font-mono text-[11px]"
              spellCheck={false}
              disabled={Boolean(busy) || loadingGraph || !editorReady}
            />
            <div className="mt-2 flex gap-1.5">
              <Button
                type="button"
                size="xs"
                variant="outline"
                onClick={() => void applyJsonDocument()}
                disabled={Boolean(busy) || loadingGraph || !editorReady}
              >
                Apply JSON
              </Button>
              {jsonDraftDirty ? (
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() => {
                    setDocumentText(pretty(graphDocument));
                    setJsonDraftDirty(false);
                  }}
                  disabled={Boolean(busy) || loadingGraph}
                >
                  Reset JSON
                </Button>
              ) : null}
            </div>
          </details>
          <div className="rounded-xl border border-border/60 bg-background/45 p-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium">Run input</p>
              <Button
                type="button"
                size="xs"
                onClick={() => void startRun()}
                disabled={
                  Boolean(busy) ||
                  loadingGraph ||
                  !editorReady ||
                  !selectedGraph ||
                  hasUnsavedGraphDraft
                }
              >
                {busy === "run" ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <Play />
                )}{" "}
                Test run
              </Button>
            </div>
            <Textarea
              value={inputText}
              onChange={(event) => {
                setInputText(event.target.value);
                setInputDraftDirty(true);
              }}
              aria-label="Graph run input JSON"
              className="mt-2 min-h-20 font-mono text-[11px]"
              spellCheck={false}
              disabled={Boolean(busy) || loadingGraph}
            />
          </div>
          {selectedRun ? (
            <div className="rounded-xl border border-border/60 bg-background/45 p-3">
              <div className="flex items-center gap-2">
                <AgentGraphLiveStatus status={selectedRun.status}>
                  <Badge variant={statusVariant(selectedRun.status)}>
                    {selectedRun.status}
                  </Badge>
                </AgentGraphLiveStatus>
                <span className="min-w-0 flex-1 truncate text-xs">
                  Run {selectedRun.id}
                </span>
                {runActionState.start ? (
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    onClick={() => void runMutation(startQueuedAgentGraphRun)}
                    aria-label="Start queued graph run"
                    disabled={Boolean(busy)}
                  >
                    <Play />
                  </Button>
                ) : null}
                {runActionState.resume ? (
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    onClick={() => void runMutation(resumeAgentGraphRun)}
                    aria-label="Resume graph run"
                    disabled={Boolean(busy)}
                  >
                    <Play />
                  </Button>
                ) : null}
                {runActionState.pause ? (
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    onClick={() => void runMutation(pauseAgentGraphRun)}
                    aria-label="Pause graph run"
                    disabled={Boolean(busy)}
                  >
                    <Pause />
                  </Button>
                ) : null}
                {runActionState.cancel ? (
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    onClick={() => void runMutation(cancelAgentGraphRun)}
                    aria-label="Cancel graph run"
                    disabled={Boolean(busy)}
                  >
                    <Square />
                  </Button>
                ) : null}
                {runActionState.retry ? (
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    onClick={() => void runMutation(retryAgentGraphRun)}
                    aria-label="Retry graph run"
                    disabled={Boolean(busy)}
                  >
                    <RotateCcw />
                  </Button>
                ) : null}
              </div>
              {selectedRun.error ? (
                <p className="mt-2 text-[11px] text-destructive">
                  {selectedRun.error}
                </p>
              ) : null}
              {activeApproval ? (
                <div className="mt-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-[11px]">
                  <p className="font-medium">{activeApproval.title}</p>
                  {activeApproval.description ? (
                    <p className="mt-1 text-muted-foreground">
                      {activeApproval.description}
                    </p>
                  ) : null}
                  <div className="mt-2 flex gap-1.5">
                    <Button
                      type="button"
                      size="xs"
                      onClick={() => void decide(activeApproval, "approved")}
                      disabled={Boolean(busy)}
                    >
                      Approve
                    </Button>
                    <Button
                      type="button"
                      size="xs"
                      variant="outline"
                      onClick={() => void decide(activeApproval, "rejected")}
                      disabled={Boolean(busy)}
                    >
                      Reject
                    </Button>
                  </div>
                </div>
              ) : null}
              <details className="mt-2" open={true}>
                <summary className="cursor-pointer text-[11px] font-medium">
                  Node executions ({runNodes.length})
                </summary>
                <div className="mt-1 space-y-1">
                  {runNodes.map((node) => (
                    <div
                      key={String(node.id)}
                      className="flex items-center gap-2 rounded bg-muted/35 px-2 py-1.5 text-[11px]"
                    >
                      <Badge variant={statusVariant(String(node.status))}>
                        {String(node.status)}
                      </Badge>
                      <span className="font-mono">{String(node.nodeId)}</span>
                    </div>
                  ))}
                </div>
              </details>
              <details className="mt-2">
                <summary className="cursor-pointer text-[11px] font-medium">
                  Event log ({runEvents.length})
                </summary>
                <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/35 p-2 font-mono text-[10px] text-muted-foreground">
                  {pretty(runEvents)}
                </pre>
              </details>
              {selectedRun.output !== null ? (
                <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/35 p-2 font-mono text-[10px]">
                  {pretty(selectedRun.output)}
                </pre>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
      {error ? (
        <p className="mt-2 text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
