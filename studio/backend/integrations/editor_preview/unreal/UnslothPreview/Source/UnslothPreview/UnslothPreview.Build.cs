// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

using UnrealBuildTool;

public class UnslothPreview : ModuleRules
{
    public UnslothPreview(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PrivateDependencyModuleNames.AddRange(new[] {
            "ApplicationCore", "Core", "CoreUObject", "Engine", "HTTP", "ImageWrapper",
            "Json", "LevelEditor", "RHI", "Slate", "SlateCore", "UnrealEd"
        });
    }
}
