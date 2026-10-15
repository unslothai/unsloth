// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Component, type ReactNode } from "react";
import { markdownBlockFallback } from "./markdown-block-fallback";

/**
 * Per-block boundary: a rejected streamdown lazy chunk rethrows during render and would otherwise
 * reach the router's catcher and replace the whole app. No retry: failed dynamic imports are cached.
 */

type Props = {
  content: string;
  children: ReactNode;
};

type State = { failed: boolean };

/**
 * Inner boundary around the renderer only, so the block's copy/download controls survive.
 * The fallback element is built by the caller: React does not catch throws in a boundary's own render.
 */
export class MarkdownRendererBoundary extends Component<
  { fallback: ReactNode; children: ReactNode },
  State
> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    console.error(
      "[markdown] a block renderer failed, showing its source instead",
      error,
    );
  }

  render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export class MarkdownBlockBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    console.error(
      "[markdown] a block failed to render, showing it as text",
      error,
    );
  }

  render(): ReactNode {
    if (!this.state.failed) {
      return this.props.children;
    }
    return <MarkdownBlockFallbackView content={this.props.content} />;
  }
}

/** Shared so both boundaries degrade a block identically (streaming fences reach the plain `Block`). */
export function MarkdownBlockFallbackView({ content }: { content: string }) {
  const fallback = markdownBlockFallback(content);
  if (fallback.fenced) {
    return (
      <div className="my-4 w-full overflow-x-auto scroll-rounded rounded-xl border border-border bg-sidebar p-2">
        {fallback.language && (
          <div className="flex h-8 items-center text-muted-foreground text-xs">
            <span className="ml-1 font-mono lowercase">
              {fallback.language}
            </span>
          </div>
        )}
        <pre className="overflow-x-auto scroll-rounded rounded-md border border-border bg-background p-4 text-sm">
          <code>{fallback.text}</code>
        </pre>
      </div>
    );
  }
  return (
    <div className="my-4 whitespace-pre-wrap break-words">{fallback.text}</div>
  );
}
