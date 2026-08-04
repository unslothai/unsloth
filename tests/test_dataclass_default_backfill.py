"""_backfill_dataclass_defaults must not shadow an inherited default.

The fix gives class-local bare annotations a `None` default so a transformers
5 config subclass survives the dataclass ordering rule. It decided "no default
yet" with `name not in cls.__dict__`, which is class-local: a subclass that
re-annotates an inherited field without assigning to it has a default already,
through the MRO, and `None` overwrote it.

Reproduced directly, so these run on transformers 4 as well -- the helper is
plain Python and needs no config machinery, unlike the live tests next door.
"""

import pytest


def test_an_inherited_default_is_not_shadowed():
    """A subclass may re-annotate an inherited field without assigning to it.
    The name is then absent from cls.__dict__ but present through the MRO, and
    writing None over it changes the config's value."""
    from unsloth.import_fixes import _backfill_dataclass_defaults

    class Base:
        window: int = 7

    class Child(Base):
        window: int          # re-annotated, no assignment

    assert _backfill_dataclass_defaults(Child) == []
    assert Child.window == 7


def test_a_genuinely_new_field_is_still_backfilled():
    """The narrowing must not disarm the fix."""
    from unsloth.import_fixes import _backfill_dataclass_defaults

    class Base:
        window: int = 7

    class Child(Base):
        vision_config: object    # new, and bare: the case that raises

    assert _backfill_dataclass_defaults(Child) == ["vision_config"]
    assert Child.vision_config is None


def test_an_inherited_method_of_the_same_name_counts_too():
    """dataclass reads defaults with getattr, so anything the MRO resolves is
    already a default, whatever it is."""
    from unsloth.import_fixes import _backfill_dataclass_defaults

    class Base:
        def helper(self):
            return 1

    class Child(Base):
        helper: object

    assert _backfill_dataclass_defaults(Child) == []
    assert Child().helper() == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
