"""Configurable LLM provider loader with environment credential management and safe fallback."""

import os
from typing import Any, List, Optional
import httpx
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from automl.core.config import Settings, get_settings
from automl.core.logging import get_logger

logger = get_logger("planner.provider")


class OpenAIChatModel(SimpleChatModel):
    """Native httpx-backed OpenAI Chat Model using langchain_core interfaces.

    Avoids external provider library bloat while providing full chat-completion capability.
    """
    model_name: str = "gpt-4o-mini"
    api_key: str
    temperature: float = 0.0
    timeout: float = 30.0

    @property
    def _llm_type(self) -> str:
        return "openai-native"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute chat completion over OpenAI compatible REST API."""
        formatted_messages = []
        for msg in messages:
            role = "user"
            if msg.type in ("system", "developer"):
                role = "system"
            elif msg.type in ("ai", "assistant"):
                role = "assistant"
            formatted_messages.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise


def get_llm_provider(
    settings: Optional[Settings] = None,
    custom_llm: Optional[Any] = None,
) -> Optional[Any]:
    """Retrieve configured LangChain chat model.

    Reads API key from environment variable OPENAI_API_KEY or application settings.
    Returns None if no API key is available or provider is unconfigured.

    Args:
        settings: Application settings.
        custom_llm: Optional custom LangChain model instance (e.g. for testing).

    Returns:
        Optional[BaseChatModel]: Configured chat model, or None if unavailable.
    """
    if custom_llm is not None:
        return custom_llm

    cfg = settings or get_settings()
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.openai_api_key

    if not api_key:
        logger.info("No LLM API key configured; LLM planning and explanation will fall back to deterministic mode.")
        return None

    # Try langchain_openai first if installed, otherwise use native OpenAIChatModel
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.llm_model_name,
            temperature=cfg.llm_temperature,
            api_key=api_key,
        )
    except ImportError:
        return OpenAIChatModel(
            model_name=cfg.llm_model_name,
            api_key=api_key,
            temperature=cfg.llm_temperature,
        )

