from unsloth.models._utils import resolve_model_class


class _AutoModelLike:
    def __init__(self, mapping):
        self._model_mapping = mapping


class _FakeLazyMapping:
    def __init__(self, entries, extra_content = None, broken_keys = ()):
        self._entries = dict(entries)
        self._config_mapping = {k: f"Cfg_{k}" for k in self._entries}
        self._model_mapping = {k: f"Mdl_{k}" for k in self._entries}
        self._extra_content = dict(extra_content or {})
        self._broken_keys = set(broken_keys)

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        for k, (cfg_cls, mdl_cls) in self._entries.items():
            if k in self._broken_keys:
                raise ValueError(f"broken entry {k}")
            if cfg_cls is key:
                return mdl_cls
        raise KeyError(key)

    def _load_attr_from_module(self, key, attr):
        if key in self._broken_keys:
            raise ValueError(f"broken entry {key}")
        cfg_cls, mdl_cls = self._entries[key]
        if attr == self._config_mapping[key]:
            return cfg_cls
        if attr == self._model_mapping[key]:
            return mdl_cls
        raise KeyError(attr)


class CfgA:
    pass


class CfgB:
    pass


class CfgBChild(CfgB):
    pass


class ModelA:
    pass


class ModelB:
    pass


class RegBase:
    pass


class RegChild(RegBase):
    pass


class RegModel:
    pass


class UnknownCfg:
    pass


def test_fast_path_exact_match():
    m = _FakeLazyMapping({"a": (CfgA, ModelA), "b": (CfgB, ModelB)})
    am = _AutoModelLike(m)
    assert resolve_model_class(am, CfgA()) is ModelA


def test_fallback_subclass_match_via_lazy_mapping():
    m = _FakeLazyMapping({"a": (CfgA, ModelA), "b": (CfgB, ModelB)})
    am = _AutoModelLike(m)
    assert resolve_model_class(am, CfgBChild()) is ModelB


def test_broken_lazy_entry_does_not_crash():
    m = _FakeLazyMapping(
        {"broken": (CfgA, ModelA), "b": (CfgB, ModelB)},
        broken_keys = ("broken",),
    )
    am = _AutoModelLike(m)
    assert resolve_model_class(am, CfgBChild()) is ModelB


def test_unknown_config_returns_none():
    m = _FakeLazyMapping({"a": (CfgA, ModelA)})
    am = _AutoModelLike(m)
    assert resolve_model_class(am, UnknownCfg()) is None


def test_extra_content_subclass_fallback():
    m = _FakeLazyMapping(
        {"a": (CfgA, ModelA)},
        extra_content = {RegBase: RegModel},
    )
    am = _AutoModelLike(m)
    assert resolve_model_class(am, RegChild()) is RegModel


def test_extra_content_exact_match_fast_path():
    m = _FakeLazyMapping(
        {"a": (CfgA, ModelA)},
        extra_content = {RegBase: RegModel},
    )
    am = _AutoModelLike(m)
    assert resolve_model_class(am, RegBase()) is RegModel


def test_broken_entry_with_extra_content_subclass():
    m = _FakeLazyMapping(
        {"broken": (CfgA, ModelA)},
        extra_content = {RegBase: RegModel},
        broken_keys = ("broken",),
    )
    am = _AutoModelLike(m)
    assert resolve_model_class(am, RegChild()) is RegModel


def test_plain_dict_mapping_is_not_required():
    am = _AutoModelLike({CfgA: ModelA})
    assert resolve_model_class(am, CfgA()) is ModelA


def test_tuple_result_unwrapped():
    m = _FakeLazyMapping({"a": (CfgA, (ModelA, "extra"))})
    am = _AutoModelLike(m)
    assert resolve_model_class(am, CfgA()) is ModelA
