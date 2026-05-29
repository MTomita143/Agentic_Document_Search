"""Azure OpenAI deployment routing helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str


def get_azure_openai_config(
    purpose: str = "fast",
    deployment: str | None = None,
) -> AzureOpenAIConfig:
    missing = missing_azure_openai_config(purpose)
    if missing:
        raise RuntimeError(
            "Azure OpenAI needs environment variables: " + ", ".join(missing)
        )

    return AzureOpenAIConfig(
        endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
        api_key=str(os.getenv("AZURE_OPENAI_API_KEY")),
        deployment=str(deployment or resolve_deployment(purpose)),
        api_version=str(
            os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION)
        ),
    )


def missing_azure_openai_config(purpose: str = "fast") -> list[str]:
    missing: list[str] = []
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        missing.append("AZURE_OPENAI_API_KEY")
    if not resolve_deployment(purpose):
        missing.append(deployment_hint(purpose))
    return missing


def resolve_deployment(purpose: str = "fast") -> str | None:
    candidates = deployment_candidates(purpose)
    return candidates[0] if candidates else None


def deployment_candidates(purpose: str = "fast") -> list[str]:
    legacy = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    fast = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT")
    deep = os.getenv("AZURE_OPENAI_DEEP_DEPLOYMENT")

    if purpose == "deep":
        return dedupe([deep, legacy, fast])
    if purpose == "fast":
        return dedupe([fast, legacy, deep])
    return dedupe([legacy, fast, deep])


def deployment_hint(purpose: str) -> str:
    if purpose == "deep":
        return (
            "AZURE_OPENAI_DEEP_DEPLOYMENT or "
            "AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_FAST_DEPLOYMENT"
        )
    if purpose == "fast":
        return "AZURE_OPENAI_FAST_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT"
    return "AZURE_OPENAI_DEPLOYMENT"


def is_deployment_not_found_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "deploymentnotfound" in text
        or "deployment for this resource does not exist" in text
        or ("404" in text and "deployment" in text)
    )


def format_deployment_not_found_message(
    purpose: str,
    attempted_deployments: list[str],
    error: Exception,
) -> str:
    return (
        "Azure OpenAI could not find the configured "
        f"{purpose} deployment. Tried: {', '.join(attempted_deployments)}. "
        "In Azure AI Foundry, open Deployments and copy the deployment name "
        "exactly; the deployment name can differ from the model name. "
        f"Original error: {error}"
    )


def dedupe(values: list[str | None]) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
