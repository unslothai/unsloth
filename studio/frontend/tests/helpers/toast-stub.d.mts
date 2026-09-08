// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Types for toast-stub.mjs so a .ts test can read the recorded calls without `any`.

export interface ToastCall {
  kind: "default" | "info" | "error" | "warning" | "success" | "dismiss";
  title?: string;
  options?: {
    id?: string;
    description?: string;
    duration?: number;
    classNames?: Record<string, string>;
  };
  id?: string;
}

export declare const calls: ToastCall[];

export declare const toast: ((title: string, options?: unknown) => void) & {
  info: (title: string, options?: unknown) => void;
  error: (title: string, options?: unknown) => void;
  warning: (title: string, options?: unknown) => void;
  success: (title: string, options?: unknown) => void;
  dismiss: (id?: string) => void;
};

export declare function createLoadingToastIcon(): null;
