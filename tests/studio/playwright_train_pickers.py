# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-browser regression coverage for the Train model and dataset pickers."""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
    is_benign_page_error,
    wait_for_health,
)

BASE = os.environ["BASE_URL"].rstrip("/")
STUDIO_PW = os.environ["STUDIO_PW"]
LOGIN_USER = os.environ.get("STUDIO_LOGIN_USER", "unsloth")
BROWSERS = tuple(
    browser.strip().lower()
    for browser in os.environ.get(
        "STUDIO_PLAYWRIGHT_BROWSERS",
        "chromium",
    ).split(",")
    if browser.strip()
)
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_train_pickers"))
TIMEOUT_MS = int(os.environ.get("STUDIO_PICKER_TIMEOUT_MS", "30000"))

LOCAL_MODEL_ALPHA = "/fixtures/models/alpha"
LOCAL_MODEL_BETA = "/fixtures/models/beta"
LOCAL_DATASET_ALPHA = "/fixtures/datasets/alpha/parquet-files/train.parquet"
LOCAL_DATASET_BETA = "/fixtures/datasets/beta/parquet-files/train.parquet"
UNMANAGED_DATASET_PATH = r"C:\fixtures\datasets\unmanaged.jsonl"
FINAL_DATASET_ID = "playwright/dataset-072"


def info(message: str) -> None:
    print(f"[train-pickers] {message}", flush = True)


def login_token() -> str:
    request = urllib.request.Request(
        f"{BASE}/api/auth/login",
        data = json.dumps(
            {"username": LOGIN_USER, "password": STUDIO_PW},
        ).encode(),
        headers = {"Content-Type": "application/json"},
        method = "POST",
    )
    with urllib.request.urlopen(request, timeout = 15) as response:
        payload = json.loads(response.read().decode())
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise AssertionError("attach-mode login returned no access_token")
    return token


def hf_model(index: int) -> dict:
    model_id = f"unsloth/playwright-model-{index:03d}"
    return {
        "_id": f"model-{index:03d}",
        "id": model_id,
        "private": False,
        "gated": False,
        "downloads": 1000 - index,
        "likes": index,
        "lastModified": "2026-01-01T00:00:00.000Z",
        "createdAt": "2025-01-01T00:00:00.000Z",
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "tags": ["transformers", "safetensors"],
        "safetensors": {
            "total": 120_000_000 + index,
            "parameters": {"F16": 120_000_000 + index},
        },
        "config": {"model_type": "llama"},
    }


def hf_dataset(index: int) -> dict:
    dataset_id = f"playwright/dataset-{index:03d}"
    return {
        "_id": f"dataset-{index:03d}",
        "id": dataset_id,
        "private": False,
        "gated": False,
        "downloads": 2000 - index,
        "likes": index,
        "lastModified": "2026-01-01T00:00:00.000Z",
        "createdAt": "2025-01-01T00:00:00.000Z",
        "tags": ["task_categories:text-generation"],
        "cardData": {
            "pretty_name": f"Playwright Dataset {index:03d}",
            "dataset_info": {
                "splits": [
                    {
                        "name": "train",
                        "num_bytes": 1024,
                        "num_examples": 10,
                    },
                ],
            },
        },
    }


LOCAL_MODELS = {
    "models_dir": "/fixtures/models",
    "hf_cache_dir": "/fixtures/hf-cache",
    "lmstudio_dirs": [],
    "ollama_dirs": [],
    "models": [
        {
            "id": "fixture-alpha",
            "load_id": LOCAL_MODEL_ALPHA,
            "display_name": "Twin Model",
            "path": LOCAL_MODEL_ALPHA,
            "source": "models_dir",
            "model_format": "safetensors",
            "runtime": "transformers",
            "capabilities": {
                "can_train": True,
                "can_chat": True,
                "can_delete": False,
                "can_download": False,
                "requires_variant": False,
                "supports_lora": True,
                "supports_vision": False,
            },
            "pipeline_tag": "text-generation",
            "tags": ["transformers", "safetensors"],
            "library_name": "transformers",
        },
        {
            "id": "fixture-beta",
            "load_id": LOCAL_MODEL_BETA,
            "display_name": "Twin Model",
            "path": LOCAL_MODEL_BETA,
            "source": "models_dir",
            "model_format": "safetensors",
            "runtime": "transformers",
            "capabilities": {
                "can_train": True,
                "can_chat": True,
                "can_delete": False,
                "can_download": False,
                "requires_variant": False,
                "supports_lora": True,
                "supports_vision": False,
            },
            "pipeline_tag": "text-generation",
            "tags": ["transformers", "safetensors"],
            "library_name": "transformers",
        },
    ],
}

LOCAL_DATASETS = {
    "datasets": [
        {
            "id": "fixture-alpha",
            "label": "Twin Dataset",
            "path": LOCAL_DATASET_ALPHA,
            "source": "recipe",
            "rows": 10,
        },
        {
            "id": "fixture-beta",
            "label": "Twin Dataset",
            "path": LOCAL_DATASET_BETA,
            "source": "upload",
            "rows": 12,
        },
    ],
}

HF_MODELS = [hf_model(index) for index in range(12)]
HF_DATASETS = [hf_dataset(index) for index in range(73)]


def fulfill_json(
    route,
    payload,
    status: int = 200,
) -> None:
    route.fulfill(
        status = status,
        body = json.dumps(payload),
        headers = {
            "access-control-allow-origin": "*",
            "cache-control": "no-store",
            "content-type": "application/json",
        },
    )


def install_route_mocks(page, counters: dict[str, int], offline: dict[str, bool]) -> None:
    page.route(
        "**/api/health*",
        lambda route: fulfill_json(
            route,
            {
                "status": "healthy",
                "service": "Unsloth UI Backend",
                "device_type": "cuda",
                "chat_only": False,
                "chat_only_reason": None,
                "secure": False,
            },
        ),
    )
    page.route(
        "**/api/hub/hidden-models*",
        lambda route: fulfill_json(
            route,
            {"needles": [], "exact_ids": [], "exact_paths": []},
        ),
    )
    page.route(
        "**/api/hub/local*",
        lambda route: fulfill_json(route, LOCAL_MODELS),
    )
    page.route(
        "**/api/hub/cached-gguf*",
        lambda route: fulfill_json(route, {"cached": []}),
    )
    page.route(
        "**/api/hub/cached-models*",
        lambda route: fulfill_json(route, {"cached": []}),
    )
    page.route(
        "**/api/hub/datasets/local*",
        lambda route: fulfill_json(route, LOCAL_DATASETS),
    )
    page.route(
        "**/api/hub/datasets/cached*",
        lambda route: fulfill_json(route, {"cached": []}),
    )

    def model_config(route) -> None:
        counters["model_config"] += 1
        model_name = urllib.parse.unquote(
            urllib.parse.urlparse(route.request.url).path.split("/config/", 1)[-1],
        )
        fulfill_json(
            route,
            {
                "id": model_name,
                "model_name": model_name,
                "config": {
                    "training": {
                        "max_seq_length": 2048,
                        "num_epochs": 1,
                        "learning_rate": 0.0002,
                    },
                },
                "is_vision": False,
                "is_embedding": False,
                "is_audio": False,
                "is_lora": False,
                "base_model": None,
                "model_type": "text",
                "max_position_embeddings": 4096,
                "model_size_bytes": 240_000_000,
            },
        )

    page.route("**/api/models/config/**", model_config)
    page.route(
        "**/api/models/check-vision/**",
        lambda route: fulfill_json(
            route,
            {"model_name": "fixture", "is_vision": False},
        ),
    )
    page.route(
        "**/api/models/check-embedding/**",
        lambda route: fulfill_json(
            route,
            {"model_name": "fixture", "is_embedding": False},
        ),
    )

    def dataset_check(route) -> None:
        counters["dataset_check"] += 1
        fulfill_json(
            route,
            {
                "requires_manual_mapping": False,
                "detected_format": "sharegpt",
                "columns": ["conversations"],
                "suggested_mapping": None,
                "preview_samples": [
                    {
                        "conversations": [
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "hi"},
                        ],
                    },
                ],
                "total_rows": 10,
                "is_image": False,
                "is_audio": False,
                "multimodal_columns": [],
                "warning": None,
            },
        )

    page.route("**/api/hub/datasets/check-format*", dataset_check)

    def hf_models(route) -> None:
        counters["hf_models"] += 1
        parsed_url = urllib.parse.urlparse(route.request.url)
        path = parsed_url.path
        if path.rstrip("/") == "/api/models":
            query = urllib.parse.parse_qs(parsed_url.query).get("search", [""])[0]
            if query:
                counters["hf_model_searches"] += 1
                normalized_query = query.casefold()
                fulfill_json(
                    route,
                    [item for item in HF_MODELS if normalized_query in item["id"].casefold()],
                )
            else:
                fulfill_json(route, HF_MODELS[:3])
            return
        requested = urllib.parse.unquote(path.split("/api/models/", 1)[-1])
        match = next((item for item in HF_MODELS if item["id"] == requested), None)
        fulfill_json(route, match or hf_model(0))

    def hf_datasets(route) -> None:
        counters["hf_datasets"] += 1
        if offline["datasets"]:
            counters["hf_dataset_failures"] += 1
            route.abort("failed")
            return
        fulfill_json(route, HF_DATASETS)

    page.route("https://huggingface.co/api/models**", hf_models)
    page.route("https://huggingface.co/api/datasets**", hf_datasets)


def training_state(page) -> dict:
    value = page.evaluate(
        """() => {
            const raw = localStorage.getItem("unsloth_training_config_v1");
            if (!raw) return {};
            try {
                const parsed = JSON.parse(raw);
                return parsed && typeof parsed.state === "object" ? parsed.state : {};
            } catch {
                return {};
            }
        }""",
    )
    return value if isinstance(value, dict) else {}


def wait_for_training_value(page, key: str, expected: str) -> None:
    page.wait_for_function(
        """({ key, expected }) => {
            const raw = localStorage.getItem("unsloth_training_config_v1");
            if (!raw) return false;
            try {
                return JSON.parse(raw)?.state?.[key] === expected;
            } catch {
                return false;
            }
        }""",
        arg = {"key": key, "expected": expected},
        timeout = TIMEOUT_MS,
    )


def open_picker(page, tour: str, noun: str):
    trigger = page.locator(f'[data-tour="{tour}"]').first
    expect(trigger).to_be_visible(timeout = TIMEOUT_MS)
    trigger.click()
    search = page.get_by_role("textbox", name = f"Search {noun}").first
    expect(search).to_be_visible(timeout = TIMEOUT_MS)
    return trigger, search


def select_picker_tab(page, name: str) -> None:
    tab = page.get_by_role(
        "tab",
        name = re.compile(rf"^\s*{re.escape(name)}\s*$", re.I),
    ).first
    expect(tab).to_be_visible(timeout = TIMEOUT_MS)
    tab.click()
    expect(tab).to_have_attribute("aria-selected", "true")


def assert_selected_picker_tab(page, name: str) -> None:
    tab = page.get_by_role(
        "tab",
        name = re.compile(rf"^\s*{re.escape(name)}\s*$", re.I),
    ).first
    expect(tab).to_have_attribute("aria-selected", "true", timeout = TIMEOUT_MS)


def close_picker(page, search) -> None:
    page.keyboard.press("Escape")
    expect(search).to_be_hidden(timeout = TIMEOUT_MS)


def assert_picker_tab_persists(page, tour: str, noun: str, tab_name: str) -> None:
    _, search = open_picker(page, tour, noun)
    tab = page.get_by_role(
        "tab",
        name = re.compile(rf"^\s*{re.escape(tab_name)}\s*$", re.I),
    ).first
    expect(tab).to_have_attribute("aria-selected", "true")
    close_picker(page, search)


def test_model_picker(page) -> None:
    info("model picker: open, arrow navigation, ambiguous keyboard match")
    model_trigger, search = open_picker(page, "studio-model-picker", "models")
    assert_selected_picker_tab(page, "On Device")
    search = page.get_by_role("textbox", name = "Search models").first
    model_options = page.locator('[data-picker-option="true"]').filter(
        has_text = "Twin Model",
    )
    expect(model_options).to_have_count(2, timeout = TIMEOUT_MS)
    search.press("ArrowDown")
    page.wait_for_function(
        """() => document.activeElement?.dataset?.pickerOption === "true" """,
        timeout = TIMEOUT_MS,
    )
    focused_values = json.loads(
        page.evaluate(
            """() => document.activeElement?.dataset?.pickerValues || "[]" """,
        ),
    )
    assert focused_values and set(focused_values) == {
        LOCAL_MODEL_ALPHA,
    }, "ArrowDown did not move from search to the first model option"
    page.keyboard.press("ArrowUp")
    expect(search).to_be_focused()

    search.fill("Twin Model")
    expect(
        page.locator('[data-picker-option="true"]').filter(has_text = "Twin Model"),
    ).to_have_count(2, timeout = TIMEOUT_MS)
    search.press("Enter")
    expect(
        search.locator("xpath=ancestor::*[@role='tabpanel']").locator("output"),
    ).to_contain_text("Multiple matching models")
    page.wait_for_function(
        """() => document.activeElement?.dataset?.pickerOption === "true" """,
        timeout = TIMEOUT_MS,
    )
    focused_values = json.loads(
        page.evaluate(
            """() => document.activeElement?.dataset?.pickerValues || "[]" """,
        ),
    )
    assert focused_values and set(focused_values) == {
        LOCAL_MODEL_ALPHA,
    }, "ambiguous model Enter did not focus the first deterministic match"

    page.keyboard.press("ArrowDown")
    focused_values = json.loads(
        page.evaluate(
            """() => document.activeElement?.dataset?.pickerValues || "[]" """,
        ),
    )
    assert focused_values and set(focused_values) == {
        LOCAL_MODEL_BETA,
    }, "ArrowDown did not move to the next model option"
    page.keyboard.press("ArrowUp")
    focused_values = json.loads(
        page.evaluate(
            """() => document.activeElement?.dataset?.pickerValues || "[]" """,
        ),
    )
    assert focused_values and set(focused_values) == {
        LOCAL_MODEL_ALPHA,
    }, "ArrowUp did not move to the previous model option"
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    expect(search).to_be_hidden(timeout = TIMEOUT_MS)
    wait_for_training_value(page, "selectedModel", LOCAL_MODEL_BETA)
    assert training_state(page).get("modelLocalPath") == LOCAL_MODEL_BETA
    expect(model_trigger).to_contain_text("Twin Model")

    info("model picker: invalid Hub ID remains open and Hub results render")
    _, search = open_picker(page, "studio-model-picker", "models")
    select_picker_tab(page, "Hugging Face")
    search = page.get_by_role("textbox", name = "Search models").first
    search.fill("bad model id!")
    search.press("Enter")
    expect(search).to_be_visible()
    assert training_state(page).get("selectedModel") == LOCAL_MODEL_BETA

    search.fill("playwright-model")
    first_result = page.locator(
        '[data-picker-option="true"][data-picker-values*="unsloth/playwright-model-011"]',
    )
    expect(first_result).to_be_visible(timeout = TIMEOUT_MS)
    close_picker(page, search)
    assert_picker_tab_persists(
        page,
        "studio-model-picker",
        "models",
        "Hugging Face",
    )


def test_dataset_picker(page) -> None:
    info("dataset picker: ambiguous exact title and exact inventory path")
    dataset_trigger, search = open_picker(
        page,
        "studio-dataset-picker",
        "datasets",
    )
    assert_selected_picker_tab(page, "On Device")
    search = page.get_by_role("textbox", name = "Search datasets").first
    search.fill("Twin Dataset")
    expect(
        page.locator('[data-picker-option="true"]').filter(
            has_text = "Twin Dataset",
        ),
    ).to_have_count(2, timeout = TIMEOUT_MS)
    search.press("Enter")
    expect(
        search.locator("xpath=ancestor::*[@role='tabpanel']").locator("output"),
    ).to_contain_text("Multiple matching datasets")
    page.wait_for_function(
        """() => document.activeElement?.dataset?.pickerOption === "true" """,
        timeout = TIMEOUT_MS,
    )
    focused_values = json.loads(
        page.evaluate(
            """() => document.activeElement?.dataset?.pickerValues || "[]" """,
        ),
    )
    assert focused_values == [
        LOCAL_DATASET_ALPHA,
    ], "ambiguous dataset Enter did not focus the first deterministic match"

    search.click()
    search.fill(LOCAL_DATASET_BETA)
    search.press("Enter")
    expect(search).to_be_hidden(timeout = TIMEOUT_MS)
    wait_for_training_value(page, "uploadedFile", LOCAL_DATASET_BETA)
    expect(dataset_trigger).to_contain_text("Twin Dataset")

    info("dataset picker: IME Enter suppression and unmanaged path rejection")
    _, search = open_picker(page, "studio-dataset-picker", "datasets")
    select_picker_tab(page, "On Device")
    search = page.get_by_role("textbox", name = "Search datasets").first
    before = training_state(page).get("uploadedFile")
    search.fill(UNMANAGED_DATASET_PATH)
    search.dispatch_event(
        "compositionstart",
        {
            "data": UNMANAGED_DATASET_PATH,
            "bubbles": True,
            "cancelable": True,
        },
    )
    search.press("Enter")
    expect(search).to_be_visible()
    assert training_state(page).get("uploadedFile") == before
    search.dispatch_event(
        "compositionend",
        {
            "data": UNMANAGED_DATASET_PATH,
            "bubbles": True,
            "cancelable": True,
        },
    )
    search.press("Enter")
    expect(search).to_be_visible()
    assert training_state(page).get("uploadedFile") == before
    close_picker(page, search)

    info("dataset picker: invalid Hub ID, 48+ result pagination, selection")
    _, search = open_picker(page, "studio-dataset-picker", "datasets")
    select_picker_tab(page, "Hugging Face")
    search = page.get_by_role("textbox", name = "Search datasets").first
    search.fill("bad dataset id!")
    search.press("Enter")
    expect(search).to_be_visible()
    assert training_state(page).get("uploadedFile") == before

    search.fill("playwright-dataset")
    first_result = page.locator(
        '[data-picker-option="true"][data-picker-values*="playwright/dataset-000"]',
    )
    expect(first_result).to_be_visible(timeout = TIMEOUT_MS)
    panel = search.locator("xpath=ancestor::*[@role='tabpanel']")
    scrollbox = panel.locator(":scope > div").last
    expect(scrollbox).to_be_visible()

    dataset_options = page.locator(
        '[data-picker-option="true"][data-picker-values*="playwright/dataset-"]',
    )
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-picker-option="true"]')]
            .filter((item) =>
                item.dataset.pickerValues?.includes("playwright/dataset-"),
            ).length >= 48
        """,
        timeout = TIMEOUT_MS,
    )
    for _ in range(4):
        scrollbox.evaluate("(element) => { element.scrollTop = element.scrollHeight; }")
        try:
            expect(dataset_options).to_have_count(73, timeout = 4000)
            break
        except AssertionError:
            continue
    expect(dataset_options).to_have_count(73, timeout = TIMEOUT_MS)

    final_result = page.locator(
        f'[data-picker-option="true"][data-picker-values*="{FINAL_DATASET_ID}"]',
    )
    final_result.scroll_into_view_if_needed()
    expect(final_result).to_be_visible()
    final_result.click()
    expect(search).to_be_hidden(timeout = TIMEOUT_MS)
    wait_for_training_value(page, "dataset", FINAL_DATASET_ID)
    state = training_state(page)
    assert state.get("datasetSource") == "huggingface"
    assert state.get("uploadedFile") is None

    assert_picker_tab_persists(
        page,
        "studio-dataset-picker",
        "datasets",
        "Hugging Face",
    )


def assert_reload_persistence(page) -> None:
    info("full reload: selected model and dataset persist")
    page.reload(wait_until = "domcontentloaded", timeout = TIMEOUT_MS)
    expect(page.locator('[data-tour="studio-model-picker"]').first).to_be_visible(
        timeout = TIMEOUT_MS,
    )
    state = training_state(page)
    assert state.get("selectedModel") == LOCAL_MODEL_BETA
    assert state.get("modelLocalPath") == LOCAL_MODEL_BETA
    assert state.get("datasetSource") == "huggingface"
    assert state.get("dataset") == FINAL_DATASET_ID
    expect(page.locator('[data-tour="studio-model-picker"]').first).to_contain_text(
        re.compile(r"beta|Twin Model", re.I),
    )
    expect(
        page.locator('[data-tour="studio-dataset-picker"]').first,
    ).to_contain_text("dataset-072")


def assert_empirical_offline(page, browser_name: str, offline: dict[str, bool]) -> None:
    info("empirical HF fetch failure drives offline state")
    assert page.evaluate("() => navigator.onLine") is True
    offline["datasets"] = True
    _, search = open_picker(page, "studio-dataset-picker", "datasets")
    select_picker_tab(page, "Hugging Face")
    search = page.get_by_role("textbox", name = "Search datasets").first
    search.fill(f"empirical-offline-{browser_name}")
    expect(page.get_by_text("You're offline", exact = True)).to_be_visible(
        timeout = TIMEOUT_MS,
    )
    expect(search).to_be_visible()


def run_browser(playwright, browser_name: str, token: str) -> None:
    info(f"{browser_name}: launch")
    browser_type = getattr(playwright, browser_name)
    launch_options: dict = {"headless": True}
    if browser_name == "chromium":
        launch_options["args"] = chromium_launch_args()
    browser = browser_type.launch(**launch_options)
    context = browser.new_context(
        viewport = {"width": 1280, "height": 900},
        reduced_motion = "reduce",
        locale = "en-US",
    )
    install_view_transition_killer(context)
    context.add_init_script(
        "try { localStorage.setItem('unsloth_auth_token', " + json.dumps(token) + "); } catch {}",
    )
    page = context.new_page()
    page.set_default_timeout(TIMEOUT_MS)
    page_errors: list[str] = []
    page.on(
        "pageerror",
        lambda error: (
            None if is_benign_page_error(str(error)) else page_errors.append(str(error))
        ),
    )
    counters = {
        "model_config": 0,
        "dataset_check": 0,
        "hf_models": 0,
        "hf_model_searches": 0,
        "hf_datasets": 0,
        "hf_dataset_failures": 0,
    }
    offline = {"datasets": False}
    install_route_mocks(page, counters, offline)

    try:
        page.goto(f"{BASE}/studio", wait_until = "domcontentloaded", timeout = TIMEOUT_MS)
        expect(page.get_by_role("tab", name = "Configure").first).to_be_visible(
            timeout = TIMEOUT_MS,
        )
        test_model_picker(page)
        test_dataset_picker(page)
        assert_reload_persistence(page)
        assert_empirical_offline(page, browser_name, offline)

        assert counters["model_config"] >= 1
        assert counters["dataset_check"] >= 3
        assert counters["hf_models"] >= 1
        assert counters["hf_model_searches"] >= 1
        assert counters["hf_datasets"] >= 1
        assert counters["hf_dataset_failures"] >= 1
        assert not page_errors, f"unexpected page errors: {page_errors}"
        page.screenshot(
            path = str(ART / f"{browser_name}-passed.png"),
            full_page = True,
            animations = "disabled",
        )
        info(f"{browser_name}: PASS")
    except Exception:
        try:
            page.screenshot(
                path = str(ART / f"{browser_name}-failed.png"),
                full_page = True,
                animations = "disabled",
            )
        except Exception:
            pass
        raise
    finally:
        context.close()
        browser.close()


def main() -> None:
    if not BROWSERS:
        raise AssertionError("STUDIO_PLAYWRIGHT_BROWSERS must name a browser")
    unsupported = sorted(set(BROWSERS) - {"chromium", "firefox", "webkit"})
    if unsupported:
        raise AssertionError(f"unsupported browser(s): {', '.join(unsupported)}")
    ART.mkdir(parents = True, exist_ok = True)
    wait_for_health(BASE, timeout = 30.0, info = info)
    token = login_token()

    failures: list[str] = []
    with sync_playwright() as playwright:
        for browser_name in BROWSERS:
            try:
                run_browser(playwright, browser_name, token)
            except Exception as error:
                failures.append(f"{browser_name}: {error}")
                info(f"{browser_name}: FAIL: {error}")

    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
