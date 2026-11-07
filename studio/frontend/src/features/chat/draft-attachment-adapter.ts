// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

const cancelledAdds = new WeakSet<File>();

// Native consumers requeue a cancelled intent; a picker add just drops with its draft.
export function wasAttachmentAddCancelled(file: File): boolean {
  return cancelledAdds.has(file);
}

// The runtime reuses its blank thread for successive drafts and clearing does not cancel adapter reads.
export class DraftScopedAttachmentAdapter implements AttachmentAdapter {
  private readonly delegate: AttachmentAdapter;
  private readonly captureCurrent: () => () => boolean;

  constructor(
    delegate: AttachmentAdapter,
    captureCurrent: () => () => boolean,
  ) {
    this.delegate = delegate;
    this.captureCurrent = captureCurrent;
  }

  get accept(): string {
    return this.delegate.accept;
  }

  add(state: { file: File }): AsyncGenerator<PendingAttachment, void> {
    // Capture at invocation, before the generator starts on its first next().
    const isCurrent = this.captureCurrent();
    cancelledAdds.delete(state.file);
    return this.addToDraft(state, isCurrent);
  }

  private async *addToDraft(
    state: { file: File },
    isCurrent: () => boolean,
  ): AsyncGenerator<PendingAttachment, void> {
    let last: PendingAttachment | undefined;
    let cancelled = false;
    const cancel = () => {
      cancelled = true;
      cancelledAdds.add(state.file);
    };
    try {
      if (!isCurrent()) {
        cancel();
        return;
      }
      const result = this.delegate.add(state);
      if (Symbol.asyncIterator in result) {
        for await (const attachment of result) {
          last = attachment;
          if (!isCurrent()) {
            cancel();
            return;
          }
          yield attachment;
        }
      } else {
        last = await result;
        if (!isCurrent()) {
          cancel();
          return;
        }
        yield last;
      }
      // Delegates return without another yield once the switch cleared their chip.
      if (!isCurrent()) cancel();
    } catch (error) {
      if (isCurrent()) throw error;
      // Throwing after a running yield would make assistant-ui resurrect the cleared chip as an error.
      cancel();
    } finally {
      if (cancelled && last) {
        // Frees adapter caches, upload slots and object URLs; removal is idempotent.
        try {
          await this.delegate.remove(last);
        } catch {
          // Cleanup must not become a stale error yield.
        }
      }
    }
  }

  remove(attachment: Attachment): Promise<void> {
    return this.delegate.remove(attachment);
  }

  send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    return this.delegate.send(attachment);
  }
}
