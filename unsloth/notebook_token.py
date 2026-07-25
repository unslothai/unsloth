# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Load the Hugging Face token exposed by hosted notebook secret stores."""

import importlib
import importlib.util
import os
from pathlib import Path


HF_TOKEN_SECRET_NAME = "HF_TOKEN"


def _read_colab_secret(name):
    if importlib.util.find_spec("google.colab") is None:
        return None
    userdata = importlib.import_module("google.colab.userdata")
    return userdata.get(name)


def _read_kaggle_secret(name):
    if importlib.util.find_spec("kaggle_secrets") is None:
        return None
    kaggle_secrets = importlib.import_module("kaggle_secrets")
    return kaggle_secrets.UserSecretsClient().get_secret(name)


def detect_notebook_hf_token(secret_name = HF_TOKEN_SECRET_NAME):
    """Populate ``HF_TOKEN`` from Colab or Kaggle, without overriding user config.

    Access failures are deliberately silent: missing/ungranted notebook secrets
    are normal, and Hugging Face can still use anonymous access or its token cache.
    """
    existing = os.environ.get("HF_TOKEN")
    if existing:
        return existing

    readers = []
    if (
        os.environ.get("COLAB_BACKEND_URL")
        or os.environ.get("COLAB_JUPYTER_IP")
        or Path("/content").is_dir()
    ):
        readers.append(_read_colab_secret)
    if (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or Path("/kaggle/working").is_dir()
    ):
        readers.append(_read_kaggle_secret)

    for reader in readers:
        try:
            token = reader(secret_name)
        except Exception:
            continue
        if isinstance(token, str) and token.strip():
            os.environ["HF_TOKEN"] = token.strip()
            return os.environ["HF_TOKEN"]
    return None
