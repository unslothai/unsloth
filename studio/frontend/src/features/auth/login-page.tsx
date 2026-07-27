// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LightRays } from "@/components/ui/light-rays";
import { Card } from "@/components/ui/card";
import { AuthForm } from "./components/auth-form";

export function LoginPage() {
  return (
    <div className="relative flex min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] items-center justify-center overflow-hidden bg-background px-4 py-8 sm:px-6 sm:py-10 md:px-10">
      <LightRays
        count={6}
        color="rgba(34, 197, 94, 0.1)"
        blur={34}
        speed={15}
        length="70vh"
        className="opacity-35 dark:opacity-15"
      />
      <Card className="relative z-10 w-full max-w-sm rounded-[2.5rem] px-7 py-8 shadow-border ring-0 sm:px-8 sm:py-10">
        <AuthForm mode="login" />
      </Card>
    </div>
  );
}
