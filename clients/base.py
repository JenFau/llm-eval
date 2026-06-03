# =============================================================================
# BASE CLASS FOR LLM CLIENTS
# =============================================================================
# This file defines a shared interface that every model client must follow.
#
# The idea: app.py shouldn't need to care which provider it's talking to.
# Whether the client connects to Groq, OpenAI, or something else, app.py
# can always call the same two methods — list_models() and stream_chat() —
# and get the same shape of result back.
#
# ABC (Abstract Base Class) enforces this contract. Any class that inherits
# from LLMClient must implement both abstract methods, or Python will raise
# an error when you try to instantiate it. This prevents accidentally shipping
# a client with a missing method.

from abc import ABC, abstractmethod
from typing import Iterator


class LLMClient(ABC):
    """Abstract base class that all model provider clients must implement."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        Return the list of model names available from this provider.
        Used to populate the model selection dropdown in the sidebar.
        """
        ...

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
    ) -> Iterator[str]:
        """
        Send a conversation to the specified model and stream the response
        back as an iterator of text chunks.

        Arguments:
            model    — the model identifier to call
            messages — conversation history as role/content dicts:
                       [{"role": "user", "content": "..."}, ...]
            system   — optional system prompt to prepend

        Yields:
            Successive text chunks as the model generates them.
        """
        ...
