# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from data_designer.plugins.plugin import Plugin, PluginType

github_repo_seed_plugin = Plugin(
    impl_qualified_name = "data_designer_github_repo_seed.impl.GitHubRepoSeedReader",
    config_qualified_name = "data_designer_github_repo_seed.config.GitHubRepoSeedSource",
    plugin_type = PluginType.SEED_READER,
)
