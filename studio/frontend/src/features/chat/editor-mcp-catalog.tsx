// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import type { EditorMcpId } from "./api/mcp-servers-api";

const strong = (text: string) => <strong className="font-medium text-foreground">{text}</strong>;

type EditorCopy = {
  name: string;
  duplicate: RegExp;
  consent: string;
  // Runs arbitrary code in the app, outside Unsloth's sandbox (Figma is read-only, Roblox Luau is Roblox-sandboxed).
  sandboxWarning?: boolean;
  link: { label: string; url: string };
  // Section of the MCP engines docs page with the app's install steps.
  guide: string;
  steps: ReactNode[];
  portHint: string;
  // Unreal: optional EngineWire project folder, read for its capability token.
  projectField?: boolean;
  // Unreal: Unsloth's level-save toolset, installed into the project (Epic's toolsets cannot save a level).
  installTools?: boolean;
  noPort?: boolean;
  more?: boolean;
};

export const EDITOR_MCP: Record<EditorMcpId, EditorCopy> = {
  unity: {
    name: "Unity",
    duplicate: /unity/i,
    consent: "Allow editing your Unity project, running menu items and C# code in the Editor. Results may be sent to your model provider. Cancelling may not undo changes.",
    link: { label: "Get MCP for Unity", url: "https://github.com/CoplayDev/unity-mcp" },
    guide: "unity",
    steps: [
      <>In Unity, open {strong("Window → Package Manager")}, click {strong("+ → Add package from git URL")} and paste <code className="break-all">https://github.com/CoplayDev/unity-mcp.git?path=/MCPForUnity#main</code>.</>,
      <>Open {strong("Window → MCP for Unity")}, keep the transport on {strong("HTTP")} and start the server.</>,
    ],
    portHint: "Match the HTTP port in MCP for Unity.",
    sandboxWarning: true,
  },
  unreal: {
    name: "Unreal Engine",
    duplicate: /unreal/i,
    consent: "Allow changing levels, assets and Blueprints and running editor commands in Unreal Editor. Results may be sent to your model provider. Cancelling may not undo changes.",
    link: { label: "Unreal MCP docs", url: "https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor" },
    guide: "unreal-engine",
    steps: [
      <>Use Unreal Engine 5.8 or newer. In {strong("Edit → Plugins")}, turn on {strong("Unreal MCP")} and {strong("All Toolsets")}, then restart the editor.</>,
      <>In {strong("Edit → Editor Preferences → General → Model Context Protocol")}, turn on {strong("Auto Start Server")} and restart. It listens on port 8000. Using the community {strong("EngineWire")} plugin instead? Set port 3000 and your project folder in Advanced settings.</>,
    ],
    portHint: "Match the Server Port Number in Editor Preferences → Model Context Protocol.",
    sandboxWarning: true,
    projectField: true,
    installTools: true,
  },
  godot: {
    name: "Godot",
    duplicate: /godot/i,
    consent: "Allow editing scenes, scripts and project files and running your project in Godot. Results may be sent to your model provider. Cancelling may not undo changes.",
    sandboxWarning: true,
    link: { label: "Get Godot MCP Native", url: "https://github.com/yurineko73/Godot-MCP-Native" },
    guide: "godot",
    steps: [
      <>In Godot 4, open the {strong("AssetLib")} tab ({strong("Asset Store")} in Godot 4.7), search {strong("Godot MCP Native")}, then download and install it.</>,
      <>Enable it in {strong("Project → Project Settings → Plugins")}.</>,
      <>Open the {strong("MCP")} tab at the top of the editor, keep the transport on {strong("HTTP")} and click {strong("Start Server")}. Tick {strong("Auto Start")} to start it with the editor next time.</>,
    ],
    portHint: "Match the Port in Godot's MCP tab.",
    more: true,
  },
  figma: {
    name: "Figma",
    duplicate: /figma/i,
    consent: "Allow reading your open Figma design files, including frames, styles and assets. Results may be sent to your model provider.",
    link: { label: "Figma MCP server guide", url: "https://help.figma.com/hc/en-us/articles/39216419318551-Get-started-with-the-Figma-MCP-server" },
    guide: "figma",
    steps: [
      <>Open a design file in the {strong("Figma desktop app")} (needs a Dev or Full seat).</>,
      <>With nothing selected, switch to {strong("Dev Mode")} and click {strong("Enable desktop MCP server")} in the right sidebar.</>,
    ],
    portHint: "The Figma desktop server uses port 3845.",
    more: true,
  },
  roblox: {
    name: "Roblox Studio",
    duplicate: /roblox/i,
    consent: "Allow reading and changing your open Roblox places and running Luau in Studio. Results may be sent to your model provider. Cancelling may not undo changes.",
    link: { label: "Roblox Studio MCP docs", url: "https://create.roblox.com/docs/studio/mcp" },
    guide: "roblox-studio",
    steps: [
      <>Open Roblox Studio on Windows or macOS.</>,
      <>Open {strong("Assistant → … → Manage MCP Servers")} and turn on {strong("Enable Studio as MCP server")}.</>,
    ],
    portHint: "Unsloth starts Roblox Studio's own MCP launcher; there is no port.",
    noPort: true,
    more: true,
  },
};
