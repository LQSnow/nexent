import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from consts.const import DEFAULT_LLM_MAX_TOKENS
from consts.model import ModelConnectStatusEnum, ModelRequest
from consts.provider import SILICON_GET_URL, ProviderEnum
from database.model_management_db import get_models_by_tenant_factory_type
from services.model_health_service import embedding_dimension_check
from utils.model_name_utils import split_repo_name, add_repo_to_name

logger = logging.getLogger("model_provider_service")


class AbstractModelProvider(ABC):
    """Common interface that all model provider integrations must implement."""

    @abstractmethod
    async def get_models(self, provider_config: Dict) -> List[Dict]:
        """Return a list of models provided by the concrete provider."""
        raise NotImplementedError


class SiliconModelProvider(AbstractModelProvider):
    """Concrete implementation for SiliconFlow provider."""

    async def get_models(self, provider_config: Dict) -> List[Dict]:
        try:
            model_type: str = provider_config["model_type"]
            model_api_key: str = provider_config["api_key"]

            headers = {"Authorization": f"Bearer {model_api_key}"}

            async with httpx.AsyncClient(verify=False) as client:
                model_list = await self._fetch_models_with_fallback(
                    client, headers, model_type
                )

            if not model_list:
                return []

            model_list = self._filter_models_by_type(model_list, model_type)

            # Annotate models with canonical fields expected downstream
            if model_type in ("llm", "vlm"):
                for item in model_list:
                    item["model_tag"] = "chat"
                    item["model_type"] = model_type
                    item.setdefault("max_tokens", DEFAULT_LLM_MAX_TOKENS)
            elif model_type in ("embedding", "multi_embedding"):
                for item in model_list:
                    item["model_tag"] = "embedding"
                    item["model_type"] = model_type

            return model_list
        except Exception as e:
            logger.error(f"Error getting models from silicon: {e}")
            return []

    async def _fetch_models_with_fallback(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        model_type: str,
    ) -> List[Dict]:
        """Fetch SiliconFlow models trying multiple parameter variants."""

        for params in self._candidate_query_params(model_type):
            silicon_url = self._build_url(params)
            try:
                response = await client.get(silicon_url, headers=headers)
                response.raise_for_status()
                payload = response.json()
                model_list = self._extract_model_list(payload)
                if model_list:
                    return model_list
            except Exception as exc:  # Broad except to log and continue fallback
                logger.warning(
                    "Failed to fetch SiliconFlow models from %s: %s",
                    silicon_url,
                    exc,
                )
                continue

        return []

    def _candidate_query_params(self, model_type: str) -> List[Optional[Dict[str, str]]]:
        """Return ordered query parameter permutations for SiliconFlow."""

        if model_type in ("llm", "vlm"):
            return [
                {"task": "chat-completion"},
                {"type": "chat"},
                {"sub_type": "chat"},
                None,
            ]
        if model_type in ("embedding", "multi_embedding"):
            return [
                {"task": "text-embedding"},
                {"type": "embedding"},
                {"sub_type": "embedding"},
                None,
            ]
        return [None]

    @staticmethod
    def _build_url(params: Optional[Dict[str, str]]) -> str:
        if not params:
            return SILICON_GET_URL
        return f"{SILICON_GET_URL}?{urlencode(params)}"

    @staticmethod
    def _extract_model_list(payload: Dict[str, Any]) -> List[Dict]:
        """Best-effort extraction of model entries from SiliconFlow payloads.

        SiliconFlow responses have changed shape multiple times. Earlier
        versions returned a simple ``{"data": [..models..]}`` payload, while
        more recent deployments wrap the list under nested ``data`` / ``list``
        / ``items`` keys. Some installations also expose the models under a top
        level ``models`` key. To keep backward compatibility we perform a
        depth-first search that stops at the first list encountered.
        """

        preferred_keys = ("data", "list", "models", "items", "records", "result", "results")

        def _search(node: Any) -> Optional[List[Dict[str, Any]]]:
            if isinstance(node, list):
                # Filter out non-dict entries to avoid iterating over primitive
                # values in subsequent processing steps.
                return [item for item in node if isinstance(item, dict)]

            if isinstance(node, dict):
                # First try well-known keys to keep the behaviour deterministic
                # when multiple lists exist at different nesting levels.
                for key in preferred_keys:
                    if key in node:
                        found = _search(node[key])
                        if found:
                            return found

                # Fall back to exploring every value. This guards against
                # undocumented schema changes where the list is stored under a
                # previously unseen key.
                for value in node.values():
                    found = _search(value)
                    if found:
                        return found

            return None

        models = _search(payload)
        return models or []

    def _filter_models_by_type(
        self, model_list: List[Dict], model_type: str
    ) -> List[Dict]:
        if not model_list:
            return model_list

        filtered = [
            model
            for model in model_list
            if self._matches_model_type(model, model_type)
        ]

        # If filtering removed everything, fall back to the original list to avoid regressions.
        return filtered or model_list

    @staticmethod
    def _matches_model_type(model: Dict[str, Any], model_type: str) -> bool:
        if model_type not in ("llm", "vlm", "embedding", "multi_embedding"):
            return True

        task = str(model.get("task", "")).lower()
        model_type_field = str(model.get("type", "")).lower()
        tags = {
            str(tag).lower()
            for tag in model.get("tags", [])
            if isinstance(tag, str)
        }

        if model_type in ("llm", "vlm"):
            llm_markers = {"chat", "chat-completion", "chat_completion", "text-generation", "vlm"}
            return bool(
                (task and any(marker in task for marker in llm_markers))
                or (model_type_field and any(marker in model_type_field for marker in llm_markers))
                or tags.intersection(llm_markers | {"llm"})
            )

        embedding_markers = {
            "embedding",
            "text-embedding",
            "text_embedding",
            "multi_embedding",
        }
        return bool(
            (task and any(marker in task for marker in embedding_markers))
            or (model_type_field and any(marker in model_type_field for marker in embedding_markers))
            or tags.intersection(embedding_markers)
        )


async def prepare_model_dict(provider: str, model: dict, model_url: str, model_api_key: str) -> dict:
    """
    Construct a model configuration dictionary that is ready to be stored in the
    database. This utility centralises the logic that was previously embedded in
    the *batch_create_models* route so that it can be reused elsewhere and keep
    the router implementation concise.

    Args:
        provider: Name of the model provider (e.g. "silicon", "openai").
        model:      A single model item coming from the provider list.
        model_url:  Base URL for the provider API.
        model_api_key: API key that should be saved together with the model.
        max_tokens: User-supplied max token / embedding dimension upper-bound.

    Returns:
        A dictionary ready to be passed to *create_model_record*.
    """

    # Split repo/name once so it can be reused multiple times.
    model_repo, model_name = split_repo_name(model["id"])
    model_display_name = add_repo_to_name(model_repo, model_name)

    # Build the canonical representation using the existing Pydantic schema for
    # consistency of validation and default handling.
    model_obj = ModelRequest(
        model_factory=provider,
        model_name=model_name,
        model_type=model["model_type"],
        api_key=model_api_key,
        max_tokens=model["max_tokens"],
        display_name=model_display_name
    )

    model_dict = model_obj.model_dump()
    model_dict["model_repo"] = model_repo or ""

    # Determine the correct base_url and, for embeddings, update the actual
    # dimension by performing a real connectivity check.
    if model["model_type"] in ["embedding", "multi_embedding"]:
        model_dict["base_url"] = f"{model_url}embeddings"
        # The embedding dimension might differ from the provided max_tokens.
        model_dict["max_tokens"] = await embedding_dimension_check(model_dict)
    else:
        model_dict["base_url"] = model_url

    # All newly created models start in NOT_DETECTED status.
    model_dict["connect_status"] = ModelConnectStatusEnum.NOT_DETECTED.value

    return model_dict


def merge_existing_model_tokens(model_list: List[dict], tenant_id: str, provider: str, model_type: str) -> List[dict]:
    """
    Merge existing model's max_tokens attribute into the model list

    Args:
        model_list: List of models
        tenant_id: Tenant ID
        provider: Provider
        model_type: Model type

    Returns:
        List[dict]: Merged model list
    """
    if model_type == "embedding" or model_type == "multi_embedding":
        return model_list

    existing_model_list = get_models_by_tenant_factory_type(
        tenant_id, provider, model_type)

    if not model_list or not existing_model_list:
        return model_list

    # Create a mapping table for existing models for quick lookup
    existing_model_map = {}
    for existing_model in existing_model_list:
        model_full_name = existing_model["model_repo"] + \
            "/" + existing_model["model_name"]
        existing_model_map[model_full_name] = existing_model

    # Iterate through the model list, if the model exists in the existing model list, add max_tokens attribute
    for model in model_list:
        if model.get("id") in existing_model_map:
            model["max_tokens"] = existing_model_map[model.get(
                "id")].get("max_tokens")

    return model_list


async def get_provider_models(model_data: dict) -> List[dict]:
    """
    Get model list based on provider

    Args:
        model_data: Model data containing provider information

    Returns:
        List[dict]: Model list
    """
    model_list = []

    if model_data["provider"] == ProviderEnum.SILICON.value:
        provider = SiliconModelProvider()
        model_list = await provider.get_models(model_data)

    return model_list
