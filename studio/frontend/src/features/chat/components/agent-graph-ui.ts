// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, createElement } from "react";

export function AgentGraphDraftStatus({
  loading,
  ready,
  pending,
  message,
}: {
  loading: boolean;
  ready: boolean;
  pending: boolean;
  message: string;
}) {
  const status = loading
    ? "loading"
    : ready
      ? pending
        ? "pending"
        : "saved"
      : "unavailable";
  const text = loading
    ? "Loading graph revision"
    : ready
      ? pending
        ? "Unsaved changes"
        : message
      : "Graph revision unavailable";
  return createElement(
    "p",
    {
      "aria-live": "polite",
      "data-agent-graph-draft-status": status,
      className: "text-[10px] text-muted-foreground",
    },
    text,
  );
}

export function AgentGraphLiveStatus({
  status,
  children,
}: {
  status: string;
  children?: ReactNode;
}) {
  return createElement(
    "span",
    {
      role: "status",
      "aria-live": "polite",
      "aria-atomic": "true",
      "data-agent-graph-run-status": status,
      className: "contents",
    },
    children ?? status,
  );
}

export function AgentGraphAlert({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return createElement(
    "p",
    {
      role: "alert",
      className,
    },
    children,
  );
}
