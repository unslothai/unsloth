// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thread switches swap the store before React unmounts old rows, so a stale index throws in
// assistant-ui 0.12 (#10288); backports assistant-ui#5119, #5127, #5948.

import type { MessageMethods, PartMethods } from "@assistant-ui/core/store";
import { AuiProvider, useAui, useAuiState } from "@assistant-ui/react";
import { Derived } from "@assistant-ui/store";
import { type FC, type PropsWithChildren, useMemo } from "react";

import {
  type TextPartState,
  createStaleSafeMessageResolver,
  textPartOrEmpty,
} from "@/components/assistant-ui/stale-safe-scope-logic";

/** `MessageByIndexProvider` whose row keeps its last message and parts while their index is past the end. */
export const MessageByIndexProvider: FC<
  PropsWithChildren<{ index: number }>
> = ({ index, children }) => {
  const resolve = useMemo(
    () => createStaleSafeMessageResolver<PartMethods, MessageMethods>(index),
    [index],
  );
  const aui = useAui({
    message: Derived({
      source: "thread",
      query: { type: "index", index },
      get: (aui) => resolve(aui.thread()),
    }),
    composer: Derived({
      source: "message",
      query: {},
      get: (aui) => resolve(aui.thread()).composer(),
    }),
  });
  return <AuiProvider value={aui}>{children}</AuiProvider>;
};

/** Composer attachment scope by id: an index scope reads a moved card's old index past the end and throws. */
export const ComposerAttachmentByIdProvider: FC<
  PropsWithChildren<{ id: string }>
> = ({ id, children }) => {
  const aui = useAui({
    attachment: Derived({
      source: "composer",
      query: { type: "id", id },
      get: (aui) => aui.composer().attachment({ id }),
    }),
  });
  return <AuiProvider value={aui}>{children}</AuiProvider>;
};

/** `useMessagePartText` that reads an empty text part instead of throwing on a mismatched part. */
export const useMessagePartTextSafe = (): TextPartState =>
  useAuiState((s) => textPartOrEmpty(s.part));
