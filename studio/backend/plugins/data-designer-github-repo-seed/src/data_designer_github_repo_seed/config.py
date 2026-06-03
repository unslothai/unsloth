# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from data_designer.config.seed_source import SeedSource


class GitHubRepoSeedSource(SeedSource):
    seed_type: Literal["github_repo"] = "github_repo"

    repos: list[str] = Field(
        default_factory = list,
        description = "List of GitHub repositories to scrape, each in `owner/name` form.",
    )
    token: str = Field(
        default = "",
        description = "Personal access token. Leave blank to read GH_TOKEN / GITHUB_TOKEN from env at run time.",
    )
    item_types: list[Literal["issues", "pulls", "commits"]] = Field(
        default = ["issues", "pulls"],
        description = "Which GitHub item types to fetch per repo.",
    )
    limit: int = Field(
        default = 100,
        ge = 1,
        le = 5000,
        description = "Maximum items per repo per item type (e.g. limit=100 + ['issues','pulls'] => up to 200 items per repo).",
    )
    include_comments: bool = Field(
        default = True,
        description = "Fetch the first N comments of each issue/PR and include them in the `comments` column.",
    )
    max_comments_per_item: int = Field(default = 30, ge = 0, le = 200)

    @field_validator("repos")
    @classmethod
    def _validate_repos(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for r in v or []:
            r = r.strip()
            if not r:
                continue
            if r.count("/") != 1 or not all(r.split("/")):
                raise ValueError(f"Each repo must be `owner/name`; got {r!r}")
            out.append(r)
        return out

    @field_validator("item_types")
    @classmethod
    def _validate_item_types(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("item_types must not be empty")
        return list(dict.fromkeys(v))

    @model_validator(mode = "after")
    def _ensure_repos(self) -> "GitHubRepoSeedSource":
        if not self.repos:
            raise ValueError("At least one repo is required")
        return self
