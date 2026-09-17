# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Run the upload scenarios through native Safari WebDriver."""

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


class SafariBrowser:
    def __init__(self, root):
        self.driver = webdriver.Safari()
        self.driver.set_script_timeout(15)
        self.version = self.driver.capabilities.get("browserVersion", "unknown")
        self.root = Path(root)

    def new_context(self):
        return SafariContext(self.driver, self.root)

    def close(self):
        self.driver.quit()


class SafariContext:
    def __init__(self, driver, root):
        self.driver, self.root = driver, root

    def new_page(self):
        return SafariPage(self.driver, self.root)

    def close(self):
        self.driver.get("about:blank")
        self.driver.delete_all_cookies()


class SafariPage:
    def __init__(self, driver, root):
        self.driver, self.root = driver, root

    def on(self, event, callback):
        # The shared fixture records page errors for Safari.
        pass

    def goto(self, url):
        self.driver.get(url)

    def evaluate(self, expression):
        result = self.driver.execute_async_script(
            "const done=arguments[arguments.length-1];"
            "Promise.resolve().then(()=>window.eval(arguments[0]))"
            ".then(value=>done({value}),error=>done({error:String(error)}));",
            expression,
        )
        if "error" in result:
            raise AssertionError(result["error"])
        return result.get("value")

    def wait_for_function(
        self,
        expression,
        timeout = 8000,
    ):
        WebDriverWait(self.driver, timeout / 1000, poll_frequency = 0.05).until(
            lambda _: self.evaluate(expression)
        )

    def wait_for_selector(self, selector):
        WebDriverWait(self.driver, 8).until(
            lambda driver: driver.find_element(By.CSS_SELECTOR, selector)
        )

    def wait_for_timeout(self, milliseconds):
        time.sleep(milliseconds / 1000)

    def locator(self, selector):
        return SafariLocator(self.driver, self.root, selector)


class SafariLocator:
    def __init__(self, driver, root, selector):
        self.driver, self.root, self.selector = driver, root, selector

    def click(self):
        self.driver.find_element(By.CSS_SELECTOR, self.selector).click()

    def set_input_files(self, file):
        directory = self.root / "safari-files"
        directory.mkdir(exist_ok = True)
        path = directory / file["name"]
        path.write_bytes(file["buffer"])
        self.driver.find_element(By.CSS_SELECTOR, self.selector).send_keys(str(path))
