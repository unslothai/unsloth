// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageTree } from "../../../i18n/types";

/**
 * Fork-only Unforgettable strings. Merged into each locale catalog at load so
 * `locales/*.ts` can track upstream without colliding on trailing namespaces
 * or settings-tab insertions.
 *
 * Non-English catalogs receive these English strings until a locale overlay
 * exists; `translate()` would also fall back to English.
 */
export const unforgettableMessages = {
  shell: {
    navigation: {
      unforgettable: "Unforgettable",
    },
  },
  settings: {
    tabs: {
      unforgettable: "Unforgettable",
    },
    unforgettable: {
      title: "Unforgettable",
      description:
        "Gated long-term notebook and rehearsal loop. Not chat RAG. Optional LoRA sidecar.",
      openDashboard: "Open Unforgettable dashboard",
      episode: {
        title: "Episode defaults",
        description:
          "Copied onto chat completions when the selected model is unforgettable.",
        planner: "Planner",
        plannerDescription: "Ask a supervisor model for a temporary plan.",
        plannerHint:
          "Working memory only. The plan is not written to the notebook.",
        plannerModel: "Planner model",
        plannerModelDescription: "Optional model id for the planner complete.",
        modelPlaceholder: "Leave blank for the inner model",
        judgeModelPlaceholder: "Leave blank for the algo",
        filter: "Filter",
        filterDescription:
          "Strip coercive and manipulative language from the user prompt.",
        filterHint:
          "A closed-list algo always runs. An LLM, if configured, may add spans.",
        filterModel: "Filter model",
        filterModelDescription: "Optional model id for the filter complete.",
        judgeModel: "Judge model",
        judgeModelDescription:
          "Optional model for holdout scoring and user-failure paraphrase.",
        judgeHint:
          "Unset keeps prefix-match eval and the closed failure-phrase list.",
        highStakes: "High stakes",
        highStakesDescription:
          "Drop sim and inferred rows from world retrieve.",
        highStakesHint:
          "Can also require confirm before retrying the world after sim.",
        confirmRetry: "Confirm world retry",
        confirmRetryDescription:
          "Show an Allow/Deny card before retrying the world.",
        confirmDefault: "Default",
        confirmAlways: "Always",
        confirmNever: "Never",
        skipStanding: "Skip standing playbooks",
        skipStandingDescription: "Do not inject compiled standing procedures.",
        adapter: "Attach adapter",
        adapterDescription: "Shrink standing for a trained sidecar adapter.",
        adapterNone: "None",
        adapterHint:
          "PEFT attaches on transformers/MLX inners. For GGUF, load with --lora and reload.",
        testCommand: "Test command",
        testCommandDescription:
          "Sim harness command. Overrides a stored test command procedure.",
        maxClones: "Max sim clones",
        maxSimTurns: "Max sim turns",
        budgetDescription:
          "Leave blank for the code default (1 clone / 8 turns).",
        twinPlugin: "Twin plugin",
        twinPluginDescription:
          "How a sim is created after a recognized failure.",
        twinFsCopy: "Filesystem copy",
        twinNone: "None (text only)",
        twinPluginHint:
          "fs.copy clones the project sandbox. none rehearses in text with no copy.",
      },
      approver: {
        title: "Approver",
        description:
          "Optional voter for admit, review, mine, compile, and promote.",
        voter: "Voter",
        voterOff: "Off",
        voterAdvisory: "Advisory",
        voterBinding: "Binding",
        voterHint: "Binding deny blocks admit and promote unless you force.",
        voterModel: "Voter model",
        supervisorUrl: "Supervisor URL",
        supervisorTimeout: "Supervisor timeout (seconds)",
      },
      store: {
        title: "Notebook",
        description: "Structured memory lives next to Studio, not in RAG.",
        path: "memory.db",
        namespace: "Namespace",
        notRag:
          "Unforgettable is not a second RAG. Chat history and rag.db are not the notebook.",
      },
    },
  },
  unforgettable: {
    page: {
      title: "Unforgettable",
      loading: "Resolving memory.db…",
      searchPlaceholder: "Search memory…",
      settings: "Settings",
    },
    inject: {
      label: "Last inject",
      standing: "standing",
      retrieve: "retrieve",
      traj: "trajectories",
      none: "No episode has written an inject split yet.",
    },
    tiles: {
      proposed: "needs review",
      active: "notebook",
      compiled: "compiled",
      archived: "archive",
      noneLive: "none live",
    },
    trust: { label: "Trust" },
    kinds: { label: "Kinds" },
    workspace: {
      inbox: "Inbox",
      notebook: "Notebook",
      standing: "Standing",
      archive: "Archive",
      sidecar: "Sidecar",
      hygiene: "Hygiene",
    },
    queue: {
      empty: "Nothing in this view.",
      askVoter: "Ask voter",
      mine: "Mine drafts",
      applyReview: "Apply review",
      applyMine: "Apply mine",
      applied: "Voter applied",
    },
    inspector: {
      noSelection: "Select a record to inspect.",
      admit: "Admit",
      reject: "Reject",
      save: "Save draft",
      force: "Force",
      compile: "Compile",
      uncompile: "Uncompile",
      deprecate: "Deprecate",
      admitted: "Admitted",
      rejected: "Rejected",
      saved: "Saved",
      compiled: "Compiled",
      uncompiled: "Removed from standing",
      deprecated: "Deprecated",
    },
    hygiene: {
      compact: "Preview compact",
      compactApply: "Apply compact",
      contradictions: "Contradictions",
      admissions: "Admissions",
      rollouts: "Rollouts",
    },
    sidecar: {
      empty: "Pack and train from the CLI. Promoted adapters appear here.",
      promote: "Promote",
      rollback: "Rollback",
      promoted: "Adapter promoted",
      rolledBack: "Adapter rolled back",
    },
    errors: {
      load: "Could not load memory",
      action: "Action failed",
      loadSettings: "Could not load Unforgettable settings",
      saveSettings: "Could not save Unforgettable settings",
    },
  },
} as const satisfies MessageTree;
