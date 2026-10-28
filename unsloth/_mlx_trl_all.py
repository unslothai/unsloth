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

"""MLX `trl.__all__`, filtered on first read (PEP 562): filtering resolves every name, importing all of trl."""

import weakref

# Kept outside trl's namespace so dir(trl) gains no bookkeeping entry.
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
        # Leave the namespace exactly as the eager filter did, so dir(trl) is unchanged.
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
            # Re-entrant read mid-filter gets the raw list, as with the eager filter.
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
        # ModuleType.__dir__ (under trl's _LazyModule.__dir__) lists the namespace: build first.
        getattr(trl, "__all__")
        return previous_dir() if previous_dir is not None else list(namespace)

    __getattr__._unsloth_previous = previous
    __dir__._unsloth_previous = previous_dir
    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
