# stt_benchmark/evaluation/config.py

"""
Evaluation config loader.

Reads a YAML eval config and produces concrete lists of:
  - ASR languages
  - AST (source, target) pairs

Supports anchor-based shorthand for AST with configurable direction:
    ast:
      anchors: [en_us, fr_fr]
      direction: both          # "forward", "reverse", or "both"

Direction controls how anchors expand:
  - "forward":  source → anchor  (source audio, anchor text)
  - "reverse":  anchor → source  (anchor audio, source text)
  - "both":     both directions
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Set


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

    # Anchors and direction
    anchors: List[str] = ast_section.get("anchors", [])
    direction: str = ast_section.get("direction", "both")

    if direction not in ("forward", "reverse", "both"):
        raise ValueError(
            f"Invalid ast.direction: '{direction}'. "
            f"Must be 'forward', 'reverse', or 'both'."
        )

    # Expand anchors based on direction
    ast_pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def _add(src: str, tgt: str):
        pair = (src, tgt)
        if src != tgt and pair not in seen:
            ast_pairs.append(pair)
            seen.add(pair)

    for src in ast_sources:
        for anchor in anchors:
            if direction in ("forward", "both"):
                _add(src, anchor)       # source audio → anchor text
            if direction in ("reverse", "both"):
                _add(anchor, src)       # anchor audio → source text

    # Extra explicit pairs (always added regardless of direction)
    extra_pairs = ast_section.get("extra_pairs", [])
    if extra_pairs:
        for entry in extra_pairs:
            _add(entry["source"], entry["target"])

    return EvalConfig(
        experiment_name=experiment_name,
        asr_languages=asr_languages,
        ast_pairs=ast_pairs,
    )