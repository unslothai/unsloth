// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function createSingleFlightRequest<TResult>(
  request: () => Promise<TResult>,
): () => Promise<TResult> {
  let inFlight: Promise<TResult> | null = null;

  return () => {
    if (inFlight) {
      return inFlight;
    }

    const current = Promise.resolve().then(request);
    inFlight = current;
    const clear = () => {
      if (inFlight === current) {
        inFlight = null;
      }
    };
    void current.then(clear, clear);
    return current;
  };
}

export interface ScopedSingleFlightRequest<TInput, TResult> {
  run: (scope: string, input: TInput) => Promise<TResult>;
  refresh: (scope: string, input: TInput) => Promise<TResult>;
}

export function createScopedSingleFlightRequest<TInput, TResult>(
  request: (
    scope: string,
    input: TInput,
    signal: AbortSignal,
  ) => Promise<TResult>,
): ScopedSingleFlightRequest<TInput, TResult> {
  let inFlight: {
    scope: string;
    controller: AbortController;
    promise: Promise<TResult>;
  } | null = null;

  const start = (scope: string, input: TInput): Promise<TResult> => {
    const controller = new AbortController();
    const promise = Promise.resolve().then(() =>
      request(scope, input, controller.signal),
    );
    const current = { scope, controller, promise };
    inFlight = current;
    const clear = () => {
      if (inFlight === current) {
        inFlight = null;
      }
    };
    void promise.then(clear, clear);
    return promise;
  };

  const supersede = (scope: string, input: TInput): Promise<TResult> => {
    inFlight?.controller.abort();
    return start(scope, input);
  };

  return {
    run: (scope, input) => {
      if (!inFlight) {
        return start(scope, input);
      }
      if (inFlight.scope === scope) {
        return inFlight.promise;
      }
      return supersede(scope, input);
    },
    refresh: supersede,
  };
}
