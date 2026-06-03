# =============================================================================
# IMPORTS
# =============================================================================
# Iterator is a type hint — it tells Python (and anyone reading the code) that
# stream_chat returns something you can loop over, one chunk at a time.
from typing import Iterator

# We use the OpenAI Python library here, not because we're talking to OpenAI,
# but because Groq's API is intentionally OpenAI-compatible. That means the
# same client library works with both — we just point it at a different URL.
# RateLimitError and BadRequestError are specific exception types we handle below.
from openai import OpenAI, RateLimitError, BadRequestError

# LLMClient is our own abstract base class (in clients/base.py). It defines the
# interface that every model client in this project must implement: list_models()
# and stream_chat(). Using a shared interface means app.py doesn't need to know
# which provider it's talking to — it just calls the same methods on any client.
from clients.base import LLMClient


# =============================================================================
# MODEL FILTER LISTS
# =============================================================================
# Groq hosts several types of models beyond text chat — audio transcription
# (Whisper), text-to-speech (PlayAI), and safety classifiers (guard models).
# These aren't useful for a text comparison tool, so we filter them out when
# building the model dropdown. We match by prefix or suffix of the model ID.
EXCLUDED_PREFIXES = ("whisper", "playai", "distil-whisper")
EXCLUDED_SUFFIXES = ("-guard",)


# =============================================================================
# CLIENT CLASS
# =============================================================================
class GroqClient(LLMClient):
    """Talks to the Groq API to list available models and stream chat responses."""

    def __init__(self, api_key: str, timeout: int = 120):
        self.timeout = timeout

        # Create an OpenAI client instance, but redirect it to Groq's servers.
        # base_url overrides the default OpenAI endpoint — this is how we reuse
        # the OpenAI library for a completely different provider.
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    # -------------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """
        Fetches the current list of available models from Groq's API.
        Returns them sorted alphabetically, excluding non-chat model types.
        Returns an empty list if the API call fails (e.g. bad key, no network).
        """
        try:
            models = self._client.models.list()
            return sorted([
                m.id for m in models.data
                # Exclude audio, speech, and safety models — text chat only
                if not any(m.id.startswith(p) for p in EXCLUDED_PREFIXES)
                and not any(m.id.endswith(s) for s in EXCLUDED_SUFFIXES)
            ])
        except Exception:
            return []

    # -------------------------------------------------------------------------
    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
    ) -> Iterator[str]:
        """
        Sends a conversation to the specified model and streams the response
        back one text chunk at a time.

        Arguments:
            model    — the Groq model ID to call (e.g. "llama-3.3-70b-versatile")
            messages — the conversation so far as a list of role/content dicts:
                       [{"role": "user", "content": "Hello"}, ...]
            system   — optional system prompt prepended before the conversation

        Yields:
            Successive text chunks as the model generates them.
            The caller concatenates these into the full response.
        """

        # Build the final message list. Chat APIs expect messages in order, and
        # the system prompt must come first, before any user/assistant messages.
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        try:
            # Ask the API to stream the response rather than waiting for the
            # entire reply to be generated before sending anything back.
            # With stream=True, the API returns a generator that yields chunks
            # as the model produces them — similar to how ChatGPT types word by word.
            stream = self._client.chat.completions.create(
                model=model,
                messages=all_messages,
                stream=True,
                timeout=self.timeout,
            )

            # Each chunk contains a small piece of the model's reply in
            # chunk.choices[0].delta.content. We yield each piece so the
            # caller can display it or accumulate it as needed.
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except RateLimitError:
            # The model is temporarily overwhelmed with requests. Surface a
            # readable message to the UI rather than a raw API exception.
            raise RuntimeError(f"**{model}** is rate-limited. Please retry in a moment.")

        except BadRequestError as e:
            # Groq returns a 400 BadRequestError when a model has been retired.
            # We catch that specific case and give a helpful message; all other
            # 400 errors are re-raised as normal so we don't hide real problems.
            if "decommissioned" in str(e):
                raise RuntimeError(f"**{model}** has been retired. Please select a different model.")
            raise
