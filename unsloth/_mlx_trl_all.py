# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The MLX `trl.__all__`, filtered on first read instead of during `import unsloth`.

Filtering resolves every exported name, and trl's lazy module imports a submodule for each,
pulling in torch, transformers, datasets and peft. Only `from trl import *`, `dir(trl)` or reading
`trl.__all__` needs the filtered list, so a PEP 562 `__getattr__` builds it then.
"""

import weakref

# trl's own names per module, kept outside its namespace so dir(trl) gains no bookkeeping entry.
_RAW_ALL = weakref.WeakKeyDictionary()


def raw_trl_all(trl):
    """trl's own export names, never resolving one."""
    if trl in _RAW_ALL:
        return list(_RAW_ALL[trl])
    return list(vars(trl).get("__all__", ()) or ())


def install_lazy_trl_all(trl, filter_exports):
    """Serve `trl.__all__` as `filter_exports(trl, names)`, computed on first access."""
    namespace = vars(trl)
    _RAW_ALL[trl] = raw_trl_all(trl)
    namespace.pop("__all__", None)
    previous = namespace.get("__getattr__")
    previous = getattr(previous, "_unsloth_previous", previous)
    previous_dir = namespace.get("__dir__")
    previous_dir = getattr(previous_dir, "_unsloth_previous", previous_dir)

    building = []

    def _hand_back():
        # Built: trl's namespace is left as the eager filter left it, so dir(trl) is unchanged.
        for key, hook, prior in (
            ("__getattr__", __getattr__, previous),
            ("__dir__", __dir__, previous_dir),
        ):
            if namespace.get(key) is hook:
                if prior is None:
                    del namespace[key]
                else:
                    namespace[key] = prior

    def __getattr__(name):
        if name == "__all__":
            raw = raw_trl_all(trl)
            # A submodule reading trl.__all__ while the filter resolves it sees the raw list,
            # as it did when the filter ran eagerly before __all__ was replaced.
            if building:
                return raw
            building.append(True)
            try:
                exports = filter_exports(trl, raw)
            finally:
                building.clear()
            namespace["__all__"] = exports
            _hand_back()
            return exports
        if previous is not None:
            return previous(name)
        # A class-level __getattr__ (trl's _LazyModule) still runs after this AttributeError.
        raise AttributeError(f"module {trl.__name__!r} has no attribute {name!r}")

    def __dir__():
        # ModuleType.__dir__ (which trl's _LazyModule.__dir__ starts from) lists the namespace:
        # build first, so a first dir(trl) already sees what the eager filter resolved.
        getattr(trl, "__all__")
        return previous_dir() if previous_dir is not None else list(namespace)

    __getattr__._unsloth_previous = previous
    __dir__._unsloth_previous = previous_dir
    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
