// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function createLibraryComposerAttachment<Item>({
  getTarget,
  getSession,
  read,
  add,
  isEditing,
}: {
  getTarget: () => string | null;
  getSession: () => number;
  read: (item: Item) => Promise<File>;
  add: (file: File) => Promise<unknown>;
  isEditing: () => boolean;
}): (item: Item) => Promise<boolean> {
  let pending = false;
  return async (item) => {
    const target = getTarget();
    const session = getSession();
    if (pending || target === null || !isEditing()) return false;
    pending = true;
    try {
      const file = await read(item);
      if (getTarget() !== target || getSession() !== session || !isEditing()) {
        return false;
      }
      await add(file);
      return true;
    } finally {
      pending = false;
    }
  };
}
