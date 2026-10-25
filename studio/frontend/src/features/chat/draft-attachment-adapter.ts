// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

const cancelledAdds = new WeakSet<File>();

// Native consumers can park the original intent after cancellation. A normal
// picker simply discards an attachment abandoned with its unsent draft.
export function wasAttachmentAddCancelled(file: File): boolean {
  return cancelledAdds.has(file);
}

/** The runtime reuses its blank thread for successive drafts. Clearing its
 * attachments does not cancel adapter reads: a late promise or generator yield
 * would otherwise put an old draft's file into the new one. */
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
    } catch (error) {
      if (isCurrent()) throw error;
      // Throwing after a running yield makes assistant-ui upsert it again with
      // error status, resurrecting a chip the new-draft switch already cleared.
      cancel();
    } finally {
      if (cancelled && last) {
        // Adapters own caches, staged upload slots and object URLs. Removal is
        // idempotent, including when the switch already removed a running chip.
        try {
          await this.delegate.remove(last);
        } catch {
          // Cleanup cannot turn cancellation into a stale error yield either.
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
