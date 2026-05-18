// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { db } from "../db";

export async function countAllChats(): Promise<number> {
  return db.threads.count();
}

export async function clearAllChats(): Promise<void> {
  await db.transaction("rw", db.threads, db.messages, async () => {
    await db.messages.clear();
    await db.threads.clear();
  });
}
