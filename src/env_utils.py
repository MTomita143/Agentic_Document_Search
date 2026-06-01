"""Environment variable helpers."""

from __future__ import annotations

import os


def clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def get_env(name: str, default: str | None = None) -> str | None:
    value = clean_env_value(os.getenv(name))
    if value is not None:
        return value
    return clean_env_value(default)


def first_env(*names: str) -> str | None:
    for name in names:
        value = get_env(name)
        if value is not None:
            return value
    return None
