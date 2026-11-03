
"""Configurable Groq LLM provider with environment credential
management and deterministic fallback."""

import os
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from automl.core.config import Settings, get_settings
from automl.core.logging import get_logger

logger = get_logger("planner.provider")


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def get_llm_provider(
    settings: Optional[Settings] = None,
    custom_llm: Optional[Any] = None,
) -> Optional[BaseChatModel]:
    """
    Retrieve the configured Groq LangChain chat model.

    Priority:
    1. Custom LLM supplied by caller.
    2. Groq API using GROQ_API_KEY.
    3. Deterministic fallback if credentials are unavailable.

    Args:
        settings: Application settings.
        custom_llm: Optional custom LangChain chat model.

    Returns:
        Configured LangChain chat model, or None if unavailable.
    """

    # Allow custom models for testing and dependency injection.
    if custom_llm is not None:
        logger.info("Using custom LLM provider.")
        return custom_llm

    cfg = settings or get_settings()

    # Read credentials from environment or application settings.
    api_key = (
        os.environ.get("GROQ_API_KEY")
        or getattr(cfg, "groq_api_key", None)
    )

    # Validate that the API key is present and not a template placeholder
    is_placeholder = bool(
        api_key
        and (
            api_key.startswith("your_")
            or "placeholder" in api_key.lower()
        )
    )

    if not api_key or is_placeholder:
        logger.warning(
            "GROQ_API_KEY not configured. "
            "LLM planning and explanation will use deterministic fallback."
        )
        return None

    # Select Groq model.
    model_name = (
        os.environ.get("GROQ_MODEL")
        or getattr(cfg, "groq_model", None)
        or getattr(cfg, "groq_model_name", None)
        or DEFAULT_GROQ_MODEL
    )

    temperature = getattr(cfg, "llm_temperature", 0.0)

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            timeout=30,
            max_retries=2,
        )

        logger.info(
            f"Groq LLM provider initialized successfully. "
            f"Model: {model_name}"
        )

        return llm

    except ImportError:
        logger.error(
            "langchain-groq is not installed. "
            "Install it using: pip install langchain-groq"
        )
        return None

    except Exception:
        logger.exception(
            "Failed to initialize Groq LLM provider. "
            "Using deterministic fallback."
        )
        return None
