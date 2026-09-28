# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Import real torchao on a torch built without torch.distributed (AMD's Windows ROCm wheels).

torchao imports DTensor / DeviceMesh / the functional collectives at module scope and keys
dispatch tables on c10d ops, none of which exist on such a build, so ``import torchao`` dies
(pytorch/ao#3452; the upstream guard, pytorch/ao#4761, is unmerged). All of it only serves
isinstance() checks and collective handlers that cannot run without distributed. So each torchao
module body executes inside a window where imports made BY torchao code get inert stand-ins for
missing torch.distributed modules and lookups made BY torchao code get inert keys for missing
c10d ops. Everything else, torch internals imported meanwhile included, sees the real (absent)
torch.distributed. Stdlib + torch only: Studio loads this file without importing unsloth.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
import types

__all__ = ["fix_torchao_without_torch_distributed"]

_OP_NAMESPACES = frozenset({"c10d_functional", "_c10d_functional", "c10d", "_dtensor"})
_IMPORTING = [0]


def _is_torchao(name):
    return isinstance(name, str) and (name == "torchao" or name.startswith("torchao."))


def _importer():
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if not name.startswith(("importlib", "_frozen_importlib")) and name != __name__:
            return name
        frame = frame.f_back
    return None


def _passthrough(fn):
    return fn


def _unavailable(qualname):
    def fn(*args, **kwargs):
        # torchao applies distributed decorators (register_sharding(op), ...) while loading;
        # there they wrap nothing. Any later call is real distributed work.
        if _IMPORTING[0]:
            if len(args) == 1 and not kwargs and isinstance(args[0], types.FunctionType):
                return args[0]
            return _passthrough
        raise RuntimeError(f"{qualname} needs torch.distributed, which this torch build lacks.")

    return fn


class _NeverMeta(type):
    def __instancecheck__(cls, obj):
        return False

    def __subclasscheck__(cls, sub):
        return False

    def __call__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls.__name__} needs torch.distributed, which this torch build lacks.")


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        name = module.__name__
        cache = {}

        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            if attr not in cache:
                cache[attr] = (
                    _NeverMeta(attr, (), {"__module__": name})
                    if attr[:1].isupper()
                    else _unavailable(f"{name}.{attr}")
                )
            return cache[attr]

        module.__getattr__ = __getattr__
        module.__path__ = []
        module.__unsloth_nodist_stub__ = True


class _DistStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if not fullname.startswith("torch.distributed.") or fullname in sys.modules:
            return None
        if not _is_torchao(_importer()):
            return None
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package = True)


class _InertPacket:
    """A missing c10d OpOverloadPacket: each overload is a unique key no real op equals."""

    def __init__(self, qualname):
        self._qualname = qualname
        self._overloads = {}

    def __getattr__(self, overload):
        if overload.startswith("__"):
            raise AttributeError(overload)
        return self._overloads.setdefault(overload, _unavailable(f"{self._qualname}.{overload}"))

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._qualname} needs torch.distributed, which this torch build lacks."
        )


class _Window:
    def __init__(self, torch):
        self.torch = torch
        self.lock = threading.RLock()
        self.depth = 0
        self.snapshots = []
        self.packets = {}
        self.dist_finder = _DistStubFinder()

    def __enter__(self):
        self.lock.acquire()
        self.depth += 1
        _IMPORTING[0] = self.depth
        self.snapshots.append(set(sys.modules))
        if self.depth == 1:
            sys.meta_path.insert(0, self.dist_finder)
            ns_cls = self.torch._ops._OpNamespace
            self.original_getattr = original = ns_cls.__getattr__
            packets = self.packets

            def __getattr__(ns, op_name):
                try:
                    return original(ns, op_name)
                except AttributeError:
                    if (
                        ns.name not in _OP_NAMESPACES
                        or op_name.startswith("__")
                        or not _is_torchao(sys._getframe(1).f_globals.get("__name__"))
                    ):
                        raise
                    key = f"{ns.name}.{op_name}"
                    return packets.setdefault(key, _InertPacket(key))

            ns_cls.__getattr__ = __getattr__
        return self

    def __exit__(self, *exc):
        try:
            self.depth -= 1
            _IMPORTING[0] = self.depth
            # This module keeps its references; nobody else may find the stand-ins.
            for name in set(sys.modules) - self.snapshots.pop():
                module = sys.modules.get(name)
                if getattr(module, "__unsloth_nodist_stub__", False):
                    del sys.modules[name]
                    parent_name, _, attr = name.rpartition(".")
                    parent = sys.modules.get(parent_name)
                    if parent is not None and getattr(parent, attr, None) is module:
                        delattr(parent, attr)
            if self.depth == 0:
                self.torch._ops._OpNamespace.__getattr__ = self.original_getattr
                sys.meta_path.remove(self.dist_finder)
        finally:
            self.lock.release()
        return False


class _WindowedLoader(importlib.abc.Loader):
    def __init__(self, loader, window):
        self.loader = loader
        self.window = window

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        with self.window:
            self.loader.exec_module(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _TorchaoWindowFinder(importlib.abc.MetaPathFinder):
    """Every torchao module, including ones transformers imports lazily later
    (quantizer_torchao -> torchao.prototype.mx_formats), loads inside the window."""

    def __init__(self, window):
        self.window = window

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if not _is_torchao(fullname):
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and not isinstance(spec.loader, _WindowedLoader):
            spec.loader = _WindowedLoader(spec.loader, self.window)
        return spec


def fix_torchao_without_torch_distributed():
    """True once real torchao is imported through the window. False, leaving nothing behind,
    where torch.distributed exists, torchao is absent or already imported (e.g. Studio's
    stub), or torchao still fails to import."""
    if any(type(f).__name__ == "_TorchaoWindowFinder" for f in sys.meta_path):
        return "torchao" in sys.modules
    if "torchao" in sys.modules:
        return False
    try:
        if importlib.util.find_spec("torchao") is None:
            return False
        import torch
        if torch.distributed.is_available():
            return False
    except Exception:
        return False

    finder = _TorchaoWindowFinder(_Window(torch))
    sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    try:
        importlib.import_module("torchao")
        return True
    except Exception:
        sys.meta_path.remove(finder)
        for name in set(sys.modules) - before:
            if _is_torchao(name):
                sys.modules.pop(name, None)
        return False
