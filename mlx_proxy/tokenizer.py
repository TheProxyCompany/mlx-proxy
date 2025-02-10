from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    A convienience wrapper around a Hugging Face tokenizer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> None:
        """
        Args:
            tokenizer: The base Hugging Face tokenizer to wrap
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._tokenizer = tokenizer

    def decode(self, tokens: list[int], **kwargs) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(tokens, **kwargs)

    def encode(
        self,
        prompt: str | list[dict[str, str]] | dict[str, str],
        apply_chat_template: bool = True,
        **kwargs,
    ) -> list[int]:
        """Encode text or chat messages into tokens.

        Handles both raw text and chat message formats. For raw text, supports
        template substitution of tools and date strings.

        Args:
            prompt: Text string or list of chat messages to encode
            **kwargs: Additional encoding options

        Returns:
            Token IDs or templated string depending on input format

        Raises:
            ValueError: If chat template produces unsupported format
        """
        if not apply_chat_template and isinstance(prompt, str):
            return self._tokenizer.encode(prompt, **kwargs)

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = prompt

        templated = self._tokenizer.apply_chat_template(messages, **kwargs)
        return templated  # type: ignore[reportReturnValue]

    @staticmethod
    def load(model_path: str | Path, **kwargs) -> Tokenizer:
        """Create a TokenizerWrapper by loading a Hugging Face tokenizer.

        Args:
            model_path: Path to the model/tokenizer
            **kwargs: Additional args passed to AutoTokenizer.from_pretrained()

        Returns:
            Configured TokenizerWrapper instance
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        return Tokenizer(tokenizer)

    def __getattribute__(self, name: str) -> Any:
        """
        Forward attribute lookups to the underlying tokenizer if not found on wrapper.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_tokenizer"), name)
