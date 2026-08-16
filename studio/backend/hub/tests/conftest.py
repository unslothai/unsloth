# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import itertools
import sys
import types


class _BaseModel:
    def __init__(self, **kwargs):
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            if name not in kwargs:
                setattr(self, name, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return dict(self.__dict__)

    def model_copy(self, update = None):
        data = self.model_dump()
        if update:
            data.update(update)
        return self.__class__(**data)


def _field(default = ..., **kwargs):
    if "default_factory" in kwargs:
        return kwargs["default_factory"]()
    return None if default is ... else default


def _model_validator(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


class _HTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        detail = None,
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _APIRouter:
    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn

    def delete(self, *args, **kwargs):
        return lambda fn: fn


def _fastapi_marker(
    default = None,
    *args,
    **kwargs,
):
    return default


class _DummyLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


sys.modules.setdefault(
    "pydantic",
    types.SimpleNamespace(
        BaseModel = _BaseModel,
        Field = _field,
        model_validator = _model_validator,
    ),
)
sys.modules.setdefault(
    "fastapi",
    types.SimpleNamespace(
        APIRouter = _APIRouter,
        Body = _fastapi_marker,
        Depends = _fastapi_marker,
        Header = _fastapi_marker,
        HTTPException = _HTTPException,
        Query = _fastapi_marker,
        UploadFile = object,
    ),
)
sys.modules.setdefault(
    "loggers",
    types.SimpleNamespace(get_logger = lambda *args, **kwargs: _DummyLogger()),
)
sys.modules.setdefault(
    "structlog",
    types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    ),
)


import pytest


@pytest.fixture(scope = "session")
def _hub_studio_home_root(tmp_path_factory):
    """One parent directory for every per-test studio home.

    ``tmp_path_factory.mktemp`` scans the whole basetemp on every call to pick
    the next number, so calling it once per test is quadratic in the number of
    tests. Paid once per session here, the per-test cost below is a bare mkdir.
    """
    return tmp_path_factory.mktemp("hub_studio_homes")


_studio_home_counter = itertools.count()


@pytest.fixture(autouse = True)
def _isolate_studio_home(_hub_studio_home_root, monkeypatch):
    home = _hub_studio_home_root / f"home-{next(_studio_home_counter)}"
    home.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("storage.", "hub.storage.")) and hasattr(module, "_schema_ready"):
            monkeypatch.setattr(module, "_schema_ready", False)


@pytest.fixture(autouse = True)
def _reset_optional_module_memo():
    """Forget the shim's memoised optional-module results between tests.

    ``_load_optional`` caches per module name including failures, so without this one test's fake
    module would answer the next test's question.
    """
    try:
        import utils.hf_xet_fallback as _shim
    except Exception:  # noqa: BLE001 - hub tests run against stubbed modules
        yield
        return
    _shim._reset_optional_module_cache()
    yield
    _shim._reset_optional_module_cache()
