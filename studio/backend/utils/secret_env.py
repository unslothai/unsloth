# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared classification for environment variables that carry credentials."""

from __future__ import annotations


SECRET_ENV_NAMES = frozenset(
    {
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "WANDB_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "REPLICATE_API_TOKEN",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "NGC_API_KEY",
        "KAGGLE_KEY",
        "MYSQL_PWD",
        "LD_PRELOAD",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "GNUPGHOME",
        "KUBECONFIG",
        "DOCKER_HOST",
    }
)
SECRET_ENV_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_", "GCP_", "GCLOUD_", "DYLD_")
SECRET_ENV_MARKERS = (
    "TOKEN",
    "API_KEY",
    "APIKEY",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PASSPHRASE",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AUTH",
    "CONNSTR",
    "CONNECTIONSTRING",
)
SECRET_ENV_KEEP_NAMES = frozenset(
    {
        "AWS_EC2_METADATA_DISABLED",
        "AWS_EC2_METADATA_V1_DISABLED",
    }
)


def is_secret_env_name(name: str) -> bool:
    """Return whether an environment variable can expose a credential."""
    upper = name.upper()
    if upper in SECRET_ENV_KEEP_NAMES:
        return False
    if upper in SECRET_ENV_NAMES:
        return True
    if any(upper.startswith(prefix) for prefix in SECRET_ENV_PREFIXES):
        return True
    return any(marker in upper for marker in SECRET_ENV_MARKERS)
