# stt_benchmark/utils/text_normalize.py

"""
Text normalization utilities for consistent metric computation.
"""

import re
from typing import Optional

class TextNormalizer:
    """Configurable text normalizer for ASR/AST evaluation."""

    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = True):
        """
        Initialize normalizer.

        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def normalize(self, text: str) -> str:
        """Apply all configured normalizations."""
        if not text:
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            # Remove punctuation but keep apostrophes within words and hyphens
            text = re.sub(r"[^\w\s\-']", "", text)
            # Clean up orphan apostrophes/hyphens
            text = re.sub(r"\s['\-]\s", " ", text)

        return text

    def get_config(self) -> dict:
        """Return normalizer configuration."""
        return {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
        }

# Default normalizer instance
DEFAULT_NORMALIZER = TextNormalizer(lowercase=True, remove_punctuation=True)