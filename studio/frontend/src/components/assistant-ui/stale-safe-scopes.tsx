// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A thread switch swaps the store to the incoming thread before React unmounts the outgoing rows, and a
// still-mounted leaf that re-renders in that window reads the new thread through its old index.
// assistant-ui 0.12 / store 0.2 throws there and the whole app becomes "Something went wrong!" (#10288):
// "tapClientLookup: Index 1 out of bounds (length: 0)" on New chat, "MessagePartText can only be used
// inside text or reasoning message parts" on opening another chat mid-reply. Upstream fixed both later
// (assistant-ui#5119, #5127, #5948); these are those fixes for the two scopes Studio owns.

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

/** `useMessagePartText` that reads an empty text part instead of throwing on a mismatched part. */
export const useMessagePartTextSafe = (): TextPartState =>
  useAuiState((s) => textPartOrEmpty(s.part));
