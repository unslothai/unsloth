// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card } from "@/components/ui/card";
import { ChatBackground } from "@/features/chat-background";
import { useLoginMode } from "./account-session";
import { AuthForm } from "./components/auth-form";

export function LoginPage() {
  // A shared sign-in screen never shows the last account's own image.
  const personal = useLoginMode() !== "multi";
  return (
    <div
      data-auth-wallpaper=""
      className="relative flex min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] items-center justify-center overflow-hidden bg-background px-4 py-8 sm:px-6 sm:py-10 md:px-10"
    >
      <ChatBackground backdrop image={personal} />
      <Card className="relative z-10 w-full max-w-sm rounded-3xl px-7 py-8 shadow-[0_4px_18px_-2px_rgba(0,0,0,0.16)] ring-0 dark:shadow-[var(--composer-shadow)] sm:px-8 sm:py-10">
        <AuthForm mode="login" />
      </Card>
    </div>
  );
}
