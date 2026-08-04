# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


@pytest.fixture(autouse = True)
def _isolate_studio_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path_factory.mktemp("studio_home")))


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
