# configs/config.py

"""
Evaluation config loader.

Reads a YAML eval config and produces concrete lists of:
  - ASR languages
  - AST (source, target) pairs

Supports anchor-based shorthand for AST:
    ast:
      anchor_targets: [en_us, fr_fr]
  expands to every (source, anchor) pair where source ≠ anchor.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set


@dataclass
class EvalConfig:
    """Parsed evaluation configuration."""
    experiment_name: str
    asr_languages: List[str]
    ast_pairs: List[Tuple[str, str]]

    def summary(self) -> str:
        lines = [
            f"Experiment: {self.experiment_name}",
            f"ASR languages: {len(self.asr_languages)}",
        ]
        if self.asr_languages:
            lines.append(f"  {', '.join(self.asr_languages)}")
        lines.append(f"AST pairs: {len(self.ast_pairs)}")
        if self.ast_pairs and len(self.ast_pairs) <= 20:
            for src, tgt in self.ast_pairs:
                lines.append(f"  {src} → {tgt}")
        elif self.ast_pairs:
            # Show first 10 + count
            for src, tgt in self.ast_pairs[:10]:
                lines.append(f"  {src} → {tgt}")
            lines.append(f"  ... and {len(self.ast_pairs) - 10} more")
        return "\n".join(lines)


def load_eval_config(config_path: str) -> EvalConfig:
    """Load and expand an evaluation config YAML.

    Args:
        config_path: Path to the eval config YAML file.

    Returns:
        EvalConfig with concrete language lists and pair lists.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    experiment_name = raw.get("experiment_name", config_path.stem)

    # ── ASR ──────────────────────────────────────────────────────────────
    asr_section = raw.get("asr", {})
    asr_languages: List[str] = asr_section.get("languages", [])

    # ── AST ──────────────────────────────────────────────────────────────
    ast_section = raw.get("ast", {})

    # Source languages: explicit list, or fall back to ASR languages
    ast_sources: List[str] = ast_section.get("sources", asr_languages)

    # Anchor targets
    anchor_targets: List[str] = ast_section.get("anchor_targets", [])

    # Expand anchors bidirectionally:
    #   source → anchor  AND  anchor → source
    ast_pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for src in ast_sources:
        for anchor in anchor_targets:
            if src != anchor:
                # source → anchor
                pair_fwd = (src, anchor)
                if pair_fwd not in seen:
                    ast_pairs.append(pair_fwd)
                    seen.add(pair_fwd)
                # anchor → source
                pair_rev = (anchor, src)
                if pair_rev not in seen:
                    ast_pairs.append(pair_rev)
                    seen.add(pair_rev)

    # Extra explicit pairs
    extra_pairs = ast_section.get("extra_pairs", [])
    if extra_pairs:
        for entry in extra_pairs:
            pair = (entry["source"], entry["target"])
            if pair not in seen:
                ast_pairs.append(pair)
                seen.add(pair)

    return EvalConfig(
        experiment_name=experiment_name,
        asr_languages=asr_languages,
        ast_pairs=ast_pairs,
    )