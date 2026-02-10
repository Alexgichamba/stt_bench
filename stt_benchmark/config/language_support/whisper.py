# stt_benchmark/config/language_support/whisper.py

"""
Whisper model language support.

Whisper uses ISO 639-1 codes for language identification.
Mapping from FLEURS codes to Whisper language tokens.
"""

from typing import Dict, Optional, Set
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES

# Whisper supported languages (ISO 639-1 codes that Whisper recognizes)
# Whisper large-v3 supports 99 languages for ASR and translation to English
WHISPER_LANGUAGES = ['ar', 'pl', 'ml', 'fo', 'mn', 'haw', 'nn', 'uz', 'cy',
                     'th', 'mr', 'sw', 'su', 'de', 'lt', 'bn', 'gl', 'sl',
                     'hi', 'am', 'fa', 'jw', 'la', 'af', 'nl', 'ht', 'ta',
                     'lb', 'uk', 'az', 'ba', 'ha', 'ps', 'br', 'is', 'ja',
                     'kk', 'so', 'sv', 'hr', 'id', 'mg', 'vi', 'tg', 'bs',
                     'ms', 'tl', 'da', 'ro', 'hy', 'zh', 'sd', 'fr', 'bo',
                     'ko', 'lo', 'si', 'es', 'tt', 'mi', 'et', 'km', 'my',
                     'no', 'hu', 'ur', 'mk', 'fi', 'el', 'tk', 'sa', 'ne',
                     'yi', 'bg', 'he', 'ca', 'sr', 'lv', 'kn', 'be', 'tr',
                     'ln', 'as', 'en', 'pa', 'yo', 'sk', 'oc', 'eu', 'yue',
                     'it', 'sn', 'gu', 'ru', 'pt', 'sq', 'ka', 'mt', 'cs', 'te']

# Whisper AST: translation is only supported TO English
WHISPER_AST_TARGET = "en"


def fleurs_to_whisper(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to Whisper language code (ISO 639-1)."""
    info = FLEURS_LANGUAGES.get(fleurs_code)
    if not info:
        return None
    iso1 = info.get("iso639_1")
    if iso1 and iso1 in WHISPER_LANGUAGES:
        return iso1
    return None

def whisper_supports_asr(fleurs_code: str) -> bool:
    """Check if Whisper supports ASR for a given FLEURS language."""
    return fleurs_to_whisper(fleurs_code) is not None

def whisper_supports_ast(source_fleurs: str, target_fleurs: str) -> bool:
    """Check if Whisper supports AST for a given language pair.
    
    Whisper only translates TO English.
    """
    if target_fleurs != "en_us":
        return False
    return fleurs_to_whisper(source_fleurs) is not None

def get_whisper_asr_languages() -> Set[str]:
    """Get FLEURS codes for all languages Whisper supports for ASR."""
    supported = set()
    for fleurs_code in FLEURS_LANGUAGES:
        if whisper_supports_asr(fleurs_code):
            supported.add(fleurs_code)
    return supported

def get_whisper_ast_pairs() -> Set[tuple]:
    """Get all (source, target) FLEURS pairs Whisper supports for AST.
    
    Returns set of (source_fleurs, 'en_us') tuples.
    """
    pairs = set()
    for fleurs_code in FLEURS_LANGUAGES:
        if fleurs_code != "en_us" and whisper_supports_asr(fleurs_code):
            pairs.add((fleurs_code, "en_us"))
    return pairs