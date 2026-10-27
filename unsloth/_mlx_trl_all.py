# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The MLX `trl.__all__`, filtered on first read instead of during `import unsloth`.

Filtering resolves every exported name, and trl's lazy module imports a submodule for each,
pulling in torch, transformers, datasets and peft. Only `from trl import *`, `dir(trl)` or reading
`trl.__all__` needs the filtered list, so a PEP 562 `__getattr__` builds it then.
"""

_RAW_ALL = "_unsloth_mlx_trl_all"


def raw_trl_all(trl):
    """trl's own export names, never resolving one."""
    namespace = vars(trl)
    if _RAW_ALL in namespace:
        return list(namespace[_RAW_ALL])
    return list(namespace.get("__all__", ()) or ())


def install_lazy_trl_all(trl, filter_exports):
    """Serve `trl.__all__` as `filter_exports(trl, names)`, computed on first access."""
    namespace = vars(trl)
    namespace[_RAW_ALL] = raw_trl_all(trl)
    namespace.pop("__all__", None)
    previous = namespace.get("__getattr__")
    previous = getattr(previous, "_unsloth_previous", previous)

    def __getattr__(name):
        if name == "__all__":
            exports = filter_exports(trl, list(vars(trl)[_RAW_ALL]))
            vars(trl)["__all__"] = exports
            return exports
        if previous is not None:
            return previous(name)
        # A class-level __getattr__ (trl's _LazyModule) still runs after this AttributeError.
        raise AttributeError(f"module {trl.__name__!r} has no attribute {name!r}")

    __getattr__._unsloth_previous = previous
    namespace["__getattr__"] = __getattr__
