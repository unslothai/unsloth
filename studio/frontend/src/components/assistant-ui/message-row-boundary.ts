// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Component, type ReactNode } from "react";

const STALE_LOOKUP = /^tapClientLookup: Index \d+ out of bounds/;
const MAX_RETRIES = 8;

type State = { error: unknown; retries: number };

/** a row that renders after a thread switch emptied the store throws tapClientLookup; this retries it next frame so TanStack Router does not replace the whole app. */
export class MessageRowBoundary extends Component<
  { children: ReactNode },
  State
> {
  state: State = { error: null, retries: 0 };
  private frame = 0;

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error };
  }

  componentDidCatch(error: unknown): void {
    if (!isStaleLookup(error) || this.state.retries >= MAX_RETRIES) {
      return;
    }
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() =>
      this.setState((state) => ({ error: null, retries: state.retries + 1 })),
    );
  }

  componentDidUpdate(): void {
    // a row that recovered starts its count again; only an error that keeps coming escalates.
    if (this.state.error == null && this.state.retries > 0) {
      this.setState({ retries: 0 });
    }
  }

  componentWillUnmount(): void {
    cancelAnimationFrame(this.frame);
  }

  render(): ReactNode {
    const { error, retries } = this.state;
    if (error == null) {
      return this.props.children;
    }
    if (isStaleLookup(error) && retries < MAX_RETRIES) {
      return null;
    }
    throw error;
  }
}

function isStaleLookup(error: unknown): boolean {
  return error instanceof Error && STALE_LOOKUP.test(error.message);
}
