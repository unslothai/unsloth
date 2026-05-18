// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// sonner `toast` wrapper that defaults `dismissible: false` so swipe
// capture doesn't block text selection. Drop-in for `from "sonner"`.

import { toast as sonnerToast, type ExternalToast } from "sonner";

type AnyFn = (...args: unknown[]) => unknown;

function withDismissibleFalse<F extends AnyFn>(fn: F): F {
  return ((...args: unknown[]) => {
    // Branch by arity: React-element messages are objects too.
    if (args.length <= 1) {
      args.push({ dismissible: false } satisfies ExternalToast);
    } else {
      const lastIdx = args.length - 1;
      const last = args[lastIdx];
      if (last && typeof last === "object" && !Array.isArray(last)) {
        const opts = last as ExternalToast;
        if (!("dismissible" in opts)) {
          args[lastIdx] = { dismissible: false, ...opts };
        }
      }
    }
    return fn(...args);
  }) as F;
}

const wrappedCallable = withDismissibleFalse(
  sonnerToast as unknown as AnyFn,
) as typeof sonnerToast;

// `promise(p, data?)` carries `dismissible` at the top of `data`,
// covering loading / success / error states. `dismiss`, `getHistory`,
// `getToasts` take no options.
const wrappedPromise: typeof sonnerToast.promise = ((promise, data) => {
  const merged =
    data && typeof data === "object" && !("dismissible" in data)
      ? { dismissible: false, ...data }
      : (data ?? { dismissible: false });
  return sonnerToast.promise(promise, merged);
}) as typeof sonnerToast.promise;

export const toast: typeof sonnerToast = Object.assign(wrappedCallable, {
  success: withDismissibleFalse(sonnerToast.success.bind(sonnerToast) as AnyFn) as typeof sonnerToast.success,
  info: withDismissibleFalse(sonnerToast.info.bind(sonnerToast) as AnyFn) as typeof sonnerToast.info,
  warning: withDismissibleFalse(sonnerToast.warning.bind(sonnerToast) as AnyFn) as typeof sonnerToast.warning,
  error: withDismissibleFalse(sonnerToast.error.bind(sonnerToast) as AnyFn) as typeof sonnerToast.error,
  message: withDismissibleFalse(sonnerToast.message.bind(sonnerToast) as AnyFn) as typeof sonnerToast.message,
  loading: withDismissibleFalse(sonnerToast.loading.bind(sonnerToast) as AnyFn) as typeof sonnerToast.loading,
  custom: withDismissibleFalse(sonnerToast.custom.bind(sonnerToast) as AnyFn) as typeof sonnerToast.custom,
  promise: wrappedPromise,
  dismiss: sonnerToast.dismiss.bind(sonnerToast) as typeof sonnerToast.dismiss,
  getHistory: sonnerToast.getHistory.bind(sonnerToast) as typeof sonnerToast.getHistory,
  getToasts: sonnerToast.getToasts.bind(sonnerToast) as typeof sonnerToast.getToasts,
});

export type { ExternalToast } from "sonner";
