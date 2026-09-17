# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared factory vocabulary for the route isolation matrix: a factory names a seeder that creates the resource inside the acting account and returns the route's path parameters, with actor expectations defaulting to the isolation contract (owner and other account 404, unauthenticated 401/403, deactivated 401)."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Callable

SEEDERS: dict[str, Callable] = {}

_CONVERTER = re.compile(r"\{([^}:]+):[^}]+\}")


def seeder(name: str):
    def decorate(function: Callable) -> Callable:
        assert name not in SEEDERS, f"duplicate seeder: {name}"
        SEEDERS[name] = function
        return function

    return decorate


def call_seeder(name: str, account, actor: str):
    function = SEEDERS[name]
    if "actor" in inspect.signature(function).parameters:
        return function(account, actor = actor)
    return function(account)


def format_path(path: str, params: dict) -> str:
    return _CONVERTER.sub(r"{\1}", path).format(**params)


@dataclass(frozen = True)
class Factory:
    name: str
    body: dict | list | None = None
    success: int = 200
    fragment: str | None = None
    absent: str | None = None
    owner: tuple[int, ...] = (404,)
    wrong: tuple[int, ...] = (404,)
    unauthenticated: tuple[int, ...] = (401, 403)
    deactivated: tuple[int, ...] = (401,)
    right: tuple[int, ...] | None = None
    self_expected: tuple[int, ...] | None = None
    reason: str = ""
    query: dict = field(default_factory = dict)
    extra_params: dict = field(default_factory = dict)

    def expected(self, actor: str) -> tuple[int, ...]:
        return {
            "owner": self.owner,
            "right": self.right or (self.success,),
            "wrong": self.wrong,
            "unauthenticated": self.unauthenticated,
            "deactivated": self.deactivated,
        }[actor]

    @property
    def deviates(self) -> bool:
        return (
            self.owner != (404,)
            or self.wrong != (404,)
            or self.right is not None
            or self.self_expected is not None
        )


def merge(*tables: dict) -> dict:
    merged: dict = {}
    for table in tables:
        for key, value in table.items():
            assert key not in merged, f"duplicate factory key: {key}"
            merged[key] = value
    return merged
