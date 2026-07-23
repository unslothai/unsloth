# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source contracts for stable compare cancellation and layout ownership."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "studio" / "frontend" / "src" / "features" / "chat"


def _read(name: str) -> str:
    return (CHAT / name).read_text()


def test_cleanup_reconciles_the_origin_checkpoint_before_clearing_it():
    composer = _read("shared-composer.tsx")
    catch = composer.split("} catch (err) {", 1)[1].split("} finally {", 1)[0]
    assert "if (run.cleanup) await run.cleanup;" in catch
    assert "const status = await getInferenceStatus();" in catch
    assert catch.index("if (run.cleanup) await run.cleanup;") < catch.index(
        "const status = await getInferenceStatus();"
    )
    assert "!run.cleanup" not in catch
    assert "(!originIsExternal && run.cleanup)" not in catch


def test_compare_layout_is_frozen_even_before_runtime_hydration():
    page = _read("chat-page.tsx")
    compare = page.split("const CompareContent = memo(", 1)[1]
    compare = compare.split("return isLoraCompare ?", 1)[0]
    assert "const [isLoraCompare] = useState(" in compare
    assert "getIsLoraCompareFromState(useChatRuntimeStore.getState())" in compare
    assert "modelRuntimeHydrated" not in compare
    assert "liveIsLoraCompare" not in compare
