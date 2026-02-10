# stt_benchmark/evaluation/metrics.py

import sacrebleu
from jiwer import wer, cer
from typing import List, Dict, Any
from stt_benchmark.evaluation.base import ASRMetrics, ASTMetrics
from stt_benchmark.utils.text_normalize import TextNormalizer, DEFAULT_NORMALIZER


class ASRMetricsCalculator:
    """Calculator for ASR metrics (WER, CER)."""
    
    def __init__(self, normalizer: TextNormalizer = None):
        """Initialize ASR metrics calculator.
        
        Args:
            normalizer: Text normalizer to apply before computing metrics.
                       Defaults to lowercase + remove punctuation + unicode normalization.
        """
        self.normalizer = normalizer or DEFAULT_NORMALIZER
    
    def calculate(self, hypotheses: List[str], references: List[str]) -> ASRMetrics:
        """Calculate corpus-level WER and CER.
        
        Args:
            hypotheses: Model outputs (transcriptions)
            references: Ground truth transcriptions
            
        Returns:
            ASRMetrics with WER and CER scores
        """
        if len(hypotheses) != len(references):
            raise ValueError(f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references")
        
        # Normalize texts
        norm_hyps = [self.normalizer.normalize(h) for h in hypotheses]
        norm_refs = [self.normalizer.normalize(r) for r in references]
        
        # Filter out empty references (can happen with bad data)
        valid_pairs = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
        if not valid_pairs:
            return ASRMetrics(wer=1.0, cer=1.0, num_samples=0, 
                            metric_config=self.normalizer.get_config())
        
        valid_hyps, valid_refs = zip(*valid_pairs)
        
        # Calculate WER and CER
        corpus_wer = wer(list(valid_refs), list(valid_hyps))
        corpus_cer = cer(list(valid_refs), list(valid_hyps))
        
        return ASRMetrics(
            wer=corpus_wer * 100,    # Convert to percentage
            cer=corpus_cer * 100,
            num_samples=len(valid_pairs),
            metric_config=self.normalizer.get_config()
        )


class ASTMetricsCalculator:
    """Calculator for AST metrics (BLEU, chrF++)."""
    
    def __init__(self, 
                 bleu_config: Dict[str, Any] = None,
                 chrf_config: Dict[str, Any] = None):
        """Initialize AST metrics calculator.
        
        Args:
            bleu_config: Config for sacrebleu BLEU (e.g., {"lowercase": False})
            chrf_config: Config for sacrebleu chrF (e.g., {"word_order": 2} for chrF++)
        """
        self.bleu_config = bleu_config or {}
        self.chrf_config = chrf_config or {"word_order": 2}
        
        self.bleu_metric = sacrebleu.BLEU(**self.bleu_config)
        self.chrf_metric = sacrebleu.CHRF(**self.chrf_config)
    
    def calculate(self, hypotheses: List[str], references: List[str]) -> ASTMetrics:
        """Calculate corpus-level BLEU and chrF++ scores.
        
        Args:
            hypotheses: Model translation outputs
            references: Ground truth translations
            
        Returns:
            ASTMetrics with BLEU and chrF++ scores
        """
        if len(hypotheses) != len(references):
            raise ValueError(f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references")
        
        bleu_score = self.bleu_metric.corpus_score(hypotheses, [references])
        chrf_score = self.chrf_metric.corpus_score(hypotheses, [references])
        
        return ASTMetrics(
            bleu=bleu_score.score,
            chrf=chrf_score.score,
            num_samples=len(hypotheses),
            metric_config={
                "bleu_config": self.bleu_config,
                "chrf_config": self.chrf_config,
            }
        )