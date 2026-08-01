// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { useHfTokenStore } from "@/features/hub";
import { useHfTokenValidation } from "@/hooks";
import { Key01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function HfTokenField() {
  const token = useHfTokenStore((state) => state.token);
  const setToken = useHfTokenStore((state) => state.setToken);
  const validation = useHfTokenValidation(token);

  return (
    <Field>
      <FieldLabel>
        Hugging Face token{" "}
        <span className="font-normal text-muted-foreground">(Optional)</span>
      </FieldLabel>
      <FieldDescription>
        Required for gated or private models and datasets.{" "}
        <a
          href="https://huggingface.co/settings/tokens"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary hover:underline"
        >
          Get token
        </a>
      </FieldDescription>
      <InputGroup>
        <InputGroupAddon>
          <HugeiconsIcon icon={Key01Icon} className="size-4" />
        </InputGroupAddon>
        <InputGroupInput
          type="password"
          autoComplete="new-password"
          spellCheck={false}
          name="hf-token"
          placeholder="hf_..."
          value={token}
          onChange={(event) => setToken(event.target.value)}
        />
      </InputGroup>
      {validation.isChecking ? (
        <p className="text-xs text-muted-foreground">Checking token…</p>
      ) : validation.error ? (
        <p className="text-xs text-destructive">{validation.error}</p>
      ) : null}
    </Field>
  );
}
