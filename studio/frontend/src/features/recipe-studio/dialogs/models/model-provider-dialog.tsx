// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useState } from "react";
import type { ModelProviderConfig } from "../../types";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type ModelProviderDialogProps = {
  config: ModelProviderConfig;
  onUpdate: (patch: Partial<ModelProviderConfig>) => void;
};

export function ModelProviderDialog({
  config,
  onUpdate,
}: ModelProviderDialogProps): ReactElement {
  const [optionalOpen, setOptionalOpen] = useState(false);
  const isLocal = config.is_local ?? false;
  const endpointId = `${config.id}-endpoint`;
  const apiKeyEnvId = `${config.id}-api-key-env`;
  const apiKeyId = `${config.id}-api-key`;
  const extraHeadersId = `${config.id}-extra-headers`;
  const extraBodyId = `${config.id}-extra-body`;
  const updateField = <K extends keyof ModelProviderConfig>(
    key: K,
    value: ModelProviderConfig[K],
  ) => {
    onUpdate({ [key]: value } as Partial<ModelProviderConfig>);
  };

  return (
    <div className="space-y-4">
      <NameField
        label="连接名称"
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />

      {/* 模型来源切换 */}
      <div className="grid gap-1.5">
        <p className="text-sm font-semibold text-foreground">模型来源</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            className={`rounded-xl border px-4 py-3 text-left transition-colors ${
              isLocal
                ? "border-primary/40 bg-primary/5"
                : "border-border/60 bg-muted/10 hover:border-border"
            }`}
            onClick={() =>
              onUpdate({
                is_local: true,
                endpoint: "",
                api_key: "",
                api_key_env: "",
                extra_headers: "",
                extra_body: "",
              })
            }
          >
            <p className="text-sm font-semibold text-foreground">
              本地模型
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              使用 Chat 页已加载的模型
            </p>
          </button>
          <button
            type="button"
            className={`rounded-xl border px-4 py-3 text-left transition-colors ${
              !isLocal
                ? "border-primary/40 bg-primary/5"
                : "border-border/60 bg-muted/10 hover:border-border"
            }`}
            onClick={() => onUpdate({ is_local: false })}
          >
            <p className="text-sm font-semibold text-foreground">
              外部端点
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              连接 OpenAI、Together 或自定义服务等 API
            </p>
          </button>
        </div>
      </div>

      {isLocal ? (
        <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
          <p className="text-sm font-semibold text-foreground">
            可以直接使用
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            点击运行时，配方会使用 Chat 页当前加载的模型，无需端点或 API 密钥。
          </p>
        </div>
      ) : (
        <>
          <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
            <p className="text-sm font-semibold text-foreground">
              先配置该模型使用的端点
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              大多数连接只需端点；若服务要求鉴权，再补充 API 密钥。
            </p>
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="端点"
              htmlFor={endpointId}
              hint="模型服务或网关的基础 URL。"
            />
            <Input
              id={endpointId}
              className="nodrag"
              placeholder="https://..."
              value={config.endpoint}
              onChange={(event) => updateField("endpoint", event.target.value)}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="API 密钥（可选）"
              htmlFor={apiKeyId}
              hint="可直接粘贴密钥，或在下方填写环境变量名。"
            />
            <Input
              id={apiKeyId}
              className="nodrag"
              value={config.api_key ?? ""}
              onChange={(event) => updateField("api_key", event.target.value)}
            />
          </div>
          <Collapsible open={optionalOpen} onOpenChange={setOptionalOpen}>
            <CollapsibleTrigger asChild={true}>
              <CollapsibleSectionTriggerButton
                label="高级请求覆写"
                open={optionalOpen}
              />
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 space-y-4">
              <div className="grid gap-1.5">
                <FieldLabel
                  label="API 密钥环境变量"
                  htmlFor={apiKeyEnvId}
                  hint="用于存放密钥的环境变量名。"
                />
                <Input
                  id={apiKeyEnvId}
                  className="nodrag"
                  placeholder="OPENAI_API_KEY"
                  value={config.api_key_env ?? ""}
                  onChange={(event) =>
                    updateField("api_key_env", event.target.value)
                  }
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="附加请求头（JSON）"
                  htmlFor={extraHeadersId}
                  hint="每次请求附带的可选请求头。"
                />
                <Textarea
                  id={extraHeadersId}
                  className="corner-squircle nodrag"
                  placeholder='{"X-Header": "value"}'
                  value={config.extra_headers ?? ""}
                  onChange={(event) =>
                    updateField("extra_headers", event.target.value)
                  }
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel
                  label="附加请求体字段（JSON）"
                  htmlFor={extraBodyId}
                  hint="每次请求附带的可选字段。"
                />
                <Textarea
                  id={extraBodyId}
                  className="corner-squircle nodrag"
                  placeholder='{"key": "value"}'
                  value={config.extra_body ?? ""}
                  onChange={(event) =>
                    updateField("extra_body", event.target.value)
                  }
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}
    </div>
  );
}
