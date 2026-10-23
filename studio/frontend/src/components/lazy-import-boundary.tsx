// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Component, type ReactNode, useEffect, useRef } from "react";

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
  const reloadRef = useRef<HTMLButtonElement>(null);
  useEffect(() => reloadRef.current?.focus(), []);
  return (
    <div role="alert" data-testid={testId} className={className}>
      <p className="text-sm">{message}</p>
      <div className="mt-2 flex gap-2">
        <button
          ref={reloadRef}
          type="button"
          onClick={() => window.location.reload()}
          className="rounded-full px-3 py-1.5 font-medium text-xs bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] hover:bg-[rgb(0_0_0_/_calc(0.09*var(--contrast-wash-gain,1)))] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
        >
          {reloadLabel}
        </button>
        {dismissLabel && onDismiss ? (
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-full px-3 py-1.5 font-medium text-xs bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] hover:bg-[rgb(0_0_0_/_calc(0.09*var(--contrast-wash-gain,1)))] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
          >
            {dismissLabel}
          </button>
        ) : null}
      </div>
    </div>
  );
}
