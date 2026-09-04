// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Component, type ReactNode } from "react";

interface LazyImportBoundaryProps {
  fallback: ReactNode;
  children: ReactNode;
}

interface LazyImportBoundaryState {
  failed: boolean;
}

export class LazyImportBoundary extends Component<
  LazyImportBoundaryProps,
  LazyImportBoundaryState
> {
  state: LazyImportBoundaryState = { failed: false };

  static getDerivedStateFromError(): LazyImportBoundaryState {
    return { failed: true };
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export function LazyImportFailure({
  message,
  reloadLabel,
  testId,
  className,
  dismissLabel,
  onDismiss,
}: {
  message: string;
  reloadLabel: string;
  testId: string;
  className: string;
  dismissLabel?: string;
  onDismiss?: () => void;
}) {
  return (
    <div role="alert" data-testid={testId} className={className}>
      <p className="text-sm">{message}</p>
      <div className="mt-2 flex gap-2">
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="rounded-md border border-border bg-background px-3 py-1.5 font-medium text-xs hover:bg-accent"
        >
          {reloadLabel}
        </button>
        {dismissLabel && onDismiss ? (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-md border border-border bg-background px-3 py-1.5 font-medium text-xs hover:bg-accent"
          >
            {dismissLabel}
          </button>
        ) : null}
      </div>
    </div>
  );
}
