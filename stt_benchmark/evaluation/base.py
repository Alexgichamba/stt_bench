# stt_benchmark/evaluation/base.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class ASRPrediction:
    """A single ASR prediction."""
    sample_id: str
    reference: str         # Ground truth transcription
    hypothesis: str        # Model output
    audio_path: str
    language: str
    prediction_time: Optional[float] = None


@dataclass
class ASTPrediction:
    """A single AST prediction."""
    sample_id: str
    source_transcription: str   # Ground truth in source language
    reference: str              # Ground truth translation in target language
    hypothesis: str             # Model output (translation)
    audio_path: str
    source_lang: str
    target_lang: str
    prediction_time: Optional[float] = None


@dataclass
class ASRMetrics:
    """Corpus-level ASR metrics."""
    wer: float               # Word Error Rate (lower is better)
    cer: float               # Character Error Rate (lower is better)
    num_samples: int
    metric_config: Dict[str, Any]


@dataclass
class ASTMetrics:
    """Corpus-level AST metrics."""
    bleu: float              # BLEU score (higher is better)
    chrf: float              # chrF++ score (higher is better)
    num_samples: int
    metric_config: Dict[str, Any]


@dataclass
class ASREvaluationResult:
    """Complete ASR evaluation result for a language."""
    language: str
    model_name: str
    predictions: List[ASRPrediction]
    metrics: ASRMetrics
    experiment_name: str
    total_time: Optional[float] = None


@dataclass
class ASTEvaluationResult:
    """Complete AST evaluation result for a language pair."""
    source_lang: str
    target_lang: str
    model_name: str
    predictions: List[ASTPrediction]
    metrics: ASTMetrics
    experiment_name: str
    total_time: Optional[float] = None