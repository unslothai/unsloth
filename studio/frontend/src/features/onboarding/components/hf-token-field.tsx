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
import { useT } from "@/i18n";
import { Key01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function HfTokenField() {
  const t = useT();
  const token = useHfTokenStore((state) => state.token);
  const setToken = useHfTokenStore((state) => state.setToken);
  const validation = useHfTokenValidation(token);

  return (
    <Field>
      <FieldLabel>
        {t("studio.wizard.hfTokenLabel")}{" "}
        <span className="font-normal text-muted-foreground">
          ({t("studio.params.optional")})
        </span>
      </FieldLabel>
      <FieldDescription>
        {t("studio.wizard.hfTokenDescription")}{" "}
        <a
          href="https://huggingface.co/settings/tokens"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary hover:underline"
        >
          {t("studio.wizard.hfTokenGet")}
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
        <p className="text-xs text-muted-foreground">
          {t("studio.wizard.hfTokenChecking")}
        </p>
      ) : validation.error ? (
        <p className="text-xs text-destructive">{validation.error}</p>
      ) : null}
    </Field>
  );
}
