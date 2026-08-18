// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Component, type ReactNode } from "react";
import { markdownBlockFallback } from "./markdown-block-fallback";

/**
 * One Markdown block that fails to render must not take the application with it.
 *
 * Streamdown renders both the syntax highlighted code body and the Mermaid
 * diagram through `React.lazy` inside its own `<Suspense>`
 * (`streamdown/dist/chunk-*.js`). The chunks are fetched the first time a reply
 * contains a fence or a diagram, so the fetch happens mid-reply, on the network
 * the app happens to have at that moment. A rejected lazy import rethrows during
 * render, and until this boundary existed the nearest catcher was TanStack
 * Router's `CatchBoundaryImpl`: one chunk that would not load replaced the whole
 * of Studio with "Something went wrong!", unmounted the assistant-ui runtime
 * with it, and left the reply's own stream with nothing consuming it. The reply
 * that was already on screen went too.
 *
 * Measured, by aborting exactly that one request on an otherwise unmodified
 * tree: the document went from 114 elements to 21, the thinking pane
 * disappeared, and the generator feeding the reply stopped two chunks later and
 * never resumed. It was also seen once with nothing injected at all.
 *
 * SCOPE. Per BLOCK, not per message, so a fence that cannot be highlighted
 * costs its own colours and nothing else: every other block in the same reply
 * keeps rendering normally. The runtime, the thread and the stream are all
 * outside and are untouched, so the reply carries on arriving.
 *
 * NO RETRY, ON PURPOSE. React and the browser's module map both cache a failed
 * dynamic import, so re-importing rethrows without issuing a new request
 * (whatwg/html#6768). `SettingsPanelBoundary` in the settings dialog reaches the
 * same conclusion for the same reason. Retrying here would throw on every frame
 * of a streaming reply for nothing. The failure is therefore sticky for the life
 * of the block, and a reader who wants the colours back reloads.
 */

type Props = {
  /** The block's Markdown source, shown as text if rendering it fails. */
  content: string;
  children: ReactNode;
};

type State = { failed: boolean };

export class MarkdownBlockBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: unknown): void {
    // Still reported. Degrading quietly is right for the READER and wrong for
    // anyone trying to find out why a reply lost its highlighting.
    console.error("[markdown] a block failed to render, showing it as text", error);
  }

  render(): ReactNode {
    if (!this.state.failed) {
      return this.props.children;
    }
    const fallback = markdownBlockFallback(this.props.content);
    if (fallback.fenced) {
      return (
        <div className="my-4 w-full overflow-x-auto rounded-xl border border-border bg-sidebar p-2">
          {fallback.language && (
            <div className="flex h-8 items-center text-muted-foreground text-xs">
              <span className="ml-1 font-mono lowercase">
                {fallback.language}
              </span>
            </div>
          )}
          <pre className="overflow-x-auto rounded-md border border-border bg-background p-4 text-sm">
            <code>{fallback.text}</code>
          </pre>
        </div>
      );
    }
    return (
      <div className="my-4 whitespace-pre-wrap break-words">{fallback.text}</div>
    );
  }
}
