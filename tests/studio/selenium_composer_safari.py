# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native Safari smoke test. Remote automation must already be enabled."""

import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import SessionNotCreatedException
from _en_catalog import en_string
from _playwright_robust import (
    open_session_with_retry,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)


def main():
    output = Path("temp/queue-validation/compatibility")
    output.mkdir(parents = True, exist_ok = True)
    server = driver = None
    report = {"browser": "native Safari", "passed": False}
    try:
        server = start_vite(5423)
        base = "http://127.0.0.1:5423"
        wait_for_smoke_page(
            base + "/smoke-prompt-queue-actions.html",
            "/smoke-prompt-queue-actions-main.tsx",
            proc = server,
        )
        driver = open_session_with_retry(webdriver.Safari, retry_on = SessionNotCreatedException)
        driver.set_window_size(1100, 900)
        report["version"] = driver.capabilities.get("browserVersion")
        wait = WebDriverWait(driver, 10)

        def label(name):
            return wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, f'[aria-label="{name}"]'))
            )

        def order(ids):
            wait.until(
                lambda d: d.execute_script(
                    "return [...document.querySelectorAll('[data-queue-item-id]')].map(row => row.dataset.queueItemId)"
                )
                == ids
            )
            driver.execute_async_script(
                "const done = arguments[0]; Promise.all([...document.querySelectorAll('[data-queue-item-id]')].flatMap(row => row.getAnimations()).map(a => a.finished.catch(() => {}))).then(done)"
            )

        def text_button(name):
            return wait.until(
                lambda d: d.find_element(By.XPATH, f'//button[normalize-space(.)="{name}"]')
            )

        def command_enter(element):
            element.click()
            ActionChains(driver).key_down(Keys.COMMAND).send_keys(Keys.ENTER).key_up(
                Keys.COMMAND
            ).perform()

        driver.get(base + "/smoke-prompt-queue-actions.html")
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "[data-queue-item-id]"))
        )
        order(["q0", "q1", "q2"])
        label("Reorder queued prompt 3 of 3").send_keys(Keys.HOME)
        order(["q2", "q0", "q1"])
        label("More options for queued prompt 1").click()
        wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//*[@role="menuitem" and contains(.,"Edit message")]')
            )
        ).click()
        editor = label("Edit queued prompt 1")
        editor.clear()
        editor.send_keys("Edited in Safari")
        command_enter(editor)
        wait.until(
            lambda d: "Edited in Safari"
            in d.find_element(By.CSS_SELECTOR, '[data-queue-item-id="q2"]').text
        )
        text_button("Reset fixture").click()
        order(["q0", "q1", "q2"])
        driver.execute_async_script(
            "const done = arguments[0]; Promise.all([...document.getAnimations()].map(a => a.finished.catch(() => {}))).then(done)"
        )
        source = label("Reorder queued prompt 3 of 3")
        target = label("Reorder queued prompt 1 of 3")
        points = driver.execute_script(
            "return [...arguments].map(el => { const r = el.getBoundingClientRect(); return {x: r.x + r.width / 2, y: r.y + r.height / 2}; })",
            source,
            target,
        )
        report["drag_points"] = points
        driver.execute_script(
            "window.queuePointerTrace = []; for (const type of ['pointerdown', 'pointermove', 'pointerup']) document.addEventListener(type, e => window.queuePointerTrace.push({type, x:e.clientX, y:e.clientY}), true)"
        )
        actions = ActionChains(driver)
        pointer = actions.w3c_actions.pointer_action
        pointer.move_to_location(round(points[0]["x"]), round(points[0]["y"]))
        pointer.pointer_down()
        pointer.move_to_location(round(points[1]["x"]), round(points[1]["y"]))
        pointer.pause(0.1)
        pointer.pointer_up()
        actions.perform()
        report["pointer_trace"] = driver.execute_script("return window.queuePointerTrace")
        order(["q2", "q0", "q1"])
        label("Steer with queued prompt 1").click()
        order(["q0", "q1"])
        assert label("Steered prompt").text == "Third prompt"
        label("Remove queued prompt 2").click()
        order(["q0"])
        driver.get(base + "/smoke-composer-settings.html")
        WebDriverWait(driver, 60).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, '[aria-label="Message"]'))
        )
        label("Message")
        driver.execute_script("localStorage.clear()")
        driver.refresh()
        label(en_string("composerSettings.plainText")).click()
        editor = label("Message")
        driver.execute_script(
            "window.composerInputTrace = []; for (const type of ['keydown', 'keyup', 'input']) document.addEventListener(type, e => window.composerInputTrace.push({type, key:e.key, meta:e.metaKey, ctrl:e.ctrlKey, value:e.target.value, label:e.target.getAttribute('aria-label')}), true)"
        )
        editor.click()
        editor.send_keys("**Safari preview**")
        wait.until(lambda d: editor.get_attribute("value") == "**Safari preview**")
        wait.until(lambda d: "Safari preview" in label("Formatted preview").text)
        label(en_string("composerSettings.showContext")).click()
        text_button("Steer").click()
        editor.click()
        editor.send_keys(Keys.ENTER)
        wait.until(
            lambda d: json.loads(label("Submitted messages").get_attribute("textContent"))
            == [{"text": "**Safari preview**", "behavior": "steer"}]
        )
        editor.send_keys("Queue once")
        command_enter(editor)
        wait.until(
            lambda d: json.loads(label("Submitted messages").get_attribute("textContent"))[-1][
                "behavior"
            ]
            == "queue"
        )
        driver.refresh()
        assert (
            label(en_string("composerSettings.plainText")).get_attribute("aria-checked") == "false"
        )
        assert (
            label(en_string("composerSettings.showContext")).get_attribute("aria-checked")
            == "false"
        )
        report["passed"] = True
        print(
            "PASS: native Safari queue controls, keyboard, pointer, preview, shortcuts and persistence",
            flush = True,
        )
    except Exception as error:
        report["error"] = str(error)
        if driver:
            report["input_trace"] = driver.execute_script("return window.composerInputTrace || []")
            driver.save_screenshot(str(output / "native-safari-failure.png"))
        raise
    finally:
        (output / "native-safari.json").write_text(
            json.dumps(report, indent = 2) + "\n", encoding = "utf-8"
        )
        if driver:
            driver.quit()
        if server:
            stop_process(server)


if __name__ == "__main__":
    main()
