# stt_benchmark/config/language_support/nllb.py

"""
NLLB (No Language Left Behind) language support for text translation.

NLLB-200 supports 200+ languages using codes in the format: {iso639_3}_{script}
Examples: swh_Latn, hau_Latn, amh_Ethi, yor_Latn

We map from FLEURS codes to NLLB codes using the iso639_3 and script fields
already present in fleurs.py.

Model variants:
- facebook/nllb-200-3.3B         (3.3B params, best quality)
- facebook/nllb-200-1.3B         (1.3B params)
- facebook/nllb-200-distilled-600M (600M params, fastest)
"""

from typing import Optional, Set, Tuple
from stt_benchmark.config.language_support.fleurs import FLEURS_LANGUAGES


# NLLB uses {iso639_3}_{script} codes. Most map directly from FLEURS metadata,
# but a few need manual overrides where NLLB uses a different ISO 639-3 code
# or a different script tag than what FLEURS lists.
NLLB_CODE_OVERRIDES = {
    # FLEURS code -> NLLB code (where automatic mapping fails)
    # Mandarin Chinese: FLEURS uses "cmn" but NLLB uses "zho_Hans"
    "cmn_hans_cn": "zho_Hans",
    # Cantonese: FLEURS uses "yue" but NLLB uses "yue_Hant"
    # (this one maps correctly automatically)

    # Norwegian Bokmål: FLEURS iso639_3 is "nob", NLLB uses "nob_Latn"
    # (maps correctly automatically)

    # Modern Standard Arabic: FLEURS uses "arb", NLLB uses "arb_Arab"
    # (maps correctly automatically)
}

# Some FLEURS languages are NOT in NLLB-200. We explicitly exclude them.
NLLB_UNSUPPORTED_FLEURS = {
    "ast_es",       # Asturian — not in NLLB
    "kea_cv",       # Kabuverdianu — not in NLLB
    "oc_fr",        # Occitan — not in NLLB
    "yue_hant_hk",  # Cantonese — not in NLLB-200 (only in some extended sets)
}


def fleurs_to_nllb(fleurs_code: str) -> Optional[str]:
    """Convert FLEURS code to NLLB language code ({iso639_3}_{script}).

    Args:
        fleurs_code: FLEURS language code (e.g., 'sw_ke', 'ha_ng')

    Returns:
        NLLB language code (e.g., 'swh_Latn', 'hau_Latn') or None if unsupported.
    """
    if fleurs_code in NLLB_UNSUPPORTED_FLEURS:
        return None

    # Check overrides first
    if fleurs_code in NLLB_CODE_OVERRIDES:
        return NLLB_CODE_OVERRIDES[fleurs_code]

    info = FLEURS_LANGUAGES.get(fleurs_code)
    if not info:
        return None

    iso3 = info.get("iso639_3")
    script = info.get("script")

    if not iso3 or not script:
        return None

    return f"{iso3}_{script}"


def nllb_supports_language(fleurs_code: str) -> bool:
    """Check if NLLB supports a FLEURS language for text translation."""
    return fleurs_to_nllb(fleurs_code) is not None


def nllb_supports_pair(source_fleurs: str, target_fleurs: str) -> bool:
    """Check if NLLB supports translating between two FLEURS languages."""
    if source_fleurs == target_fleurs:
        return False
    return (nllb_supports_language(source_fleurs) and
            nllb_supports_language(target_fleurs))


def get_nllb_supported_languages() -> Set[str]:
    """Get FLEURS codes for all languages NLLB supports."""
    supported = set()
    for fleurs_code in FLEURS_LANGUAGES:
        if nllb_supports_language(fleurs_code):
            supported.add(fleurs_code)
    return supported


def get_nllb_supported_pairs(anchor_targets: Set[str] = None) -> Set[Tuple[str, str]]:
    """Get supported translation pairs.

    Args:
        anchor_targets: If provided, only return pairs involving these as targets.
                       Defaults to all-to-all.
    """
    supported_langs = get_nllb_supported_languages()

    if anchor_targets is None:
        anchor_targets = supported_langs

    pairs = set()
    for src in supported_langs:
        for tgt in anchor_targets:
            if src != tgt and tgt in supported_langs:
                pairs.add((src, tgt))
    return pairs