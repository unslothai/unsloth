// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import type { EditorMcpId } from "./api/mcp-servers-api";

const strong = (text: string) => <strong className="font-medium text-foreground">{text}</strong>;

type EditorCopy = {
  name: string;
  duplicate: RegExp;
  consent: string;
  link: { label: string; url: string };
  steps: ReactNode[];
  portHint: string;
  needsProject?: boolean;
};

export const EDITOR_MCP: Record<EditorMcpId, EditorCopy> = {
  unity: {
    name: "Unity",
    duplicate: /unity/i,
    consent: "Allow editing your Unity project, running menu items and C# code in the Editor. Results may be sent to your model provider. Cancelling may not undo changes.",
    link: { label: "Get MCP for Unity", url: "https://github.com/CoplayDev/unity-mcp" },
    steps: [
      <>In Unity, open {strong("Window → Package Manager")}, click {strong("+ → Add package from git URL")} and paste <code className="break-all">https://github.com/CoplayDev/unity-mcp.git?path=/MCPForUnity#main</code>.</>,
      <>Open {strong("Window → MCP for Unity")}, keep the transport on {strong("HTTP")} and start the server.</>,
    ],
    portHint: "Match the HTTP port in MCP for Unity.",
  },
  unreal: {
    name: "Unreal Engine",
    duplicate: /unreal/i,
    consent: "Allow changing levels, assets and Blueprints and running editor commands in Unreal Editor. Results may be sent to your model provider. Cancelling may not undo changes.",
    link: { label: "Get Unreal MCP plugin", url: "https://github.com/ChiR24/Unreal_mcp" },
    steps: [
      <>Install the {strong("EngineWire")} plugin from Fab (or copy it into your project’s {strong("Plugins")} folder), enable it in {strong("Edit → Plugins")} and restart the editor.</>,
      <>If the status bar reads {strong("MCP off")}, turn on {strong("Enable Native MCP Server")} in {strong("Project Settings → Plugins → EngineWire")} and restart once more.</>,
      <>Enter your project folder above so Unsloth can read the plugin’s capability token.</>,
    ],
    portHint: "Match the Native MCP Server port in the plugin settings.",
    needsProject: true,
  },
  godot: {
    name: "Godot",
    duplicate: /godot/i,
    consent: "Allow editing scenes, scripts and project files and running your project in Godot. Results may be sent to your model provider. Cancelling may not undo changes.",
    link: { label: "Get Godot MCP Native", url: "https://github.com/yurineko73/Godot-MCP-Native" },
    steps: [
      <>In Godot 4, open the {strong("AssetLib")} tab, search {strong("Godot MCP Native")}, then download and install it.</>,
      <>Enable it in {strong("Project → Project Settings → Plugins")}.</>,
      <>In the {strong("MCP")} dock, keep the transport on {strong("HTTP")} and start the server if it is not running.</>,
    ],
    portHint: "Match the HTTP port in the MCP dock.",
  },
};
