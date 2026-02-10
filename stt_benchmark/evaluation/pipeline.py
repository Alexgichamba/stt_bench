# stt_benchmark/evaluation/pipeline.py

import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from stt_benchmark.models.base import BaseASRModel, BaseASTModel
from stt_benchmark.datasets.base import BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample
from stt_benchmark.evaluation.base import (
    ASRPrediction, ASTPrediction, ASREvaluationResult, ASTEvaluationResult
)
from stt_benchmark.evaluation.metrics import ASRMetricsCalculator, ASTMetricsCalculator
from stt_benchmark.utils.audio import load_audio
from stt_benchmark.utils.text_normalize import TextNormalizer


class EvaluationPipeline:
    """Main evaluation pipeline for ASR and AST models."""
    
    def __init__(self,
                 output_dir: str = "results",
                 batch_size: int = 1,
                 target_sr: int = 16000,
                 normalizer: TextNormalizer = None,
                 bleu_config: Dict[str, Any] = None,
                 chrf_config: Dict[str, Any] = None,
                 skip_unsupported: bool = True):
        """Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save results
            batch_size: Batch size for inference
            target_sr: Target audio sampling rate
            normalizer: Text normalizer for ASR WER computation
            bleu_config: BLEU config for AST
            chrf_config: chrF config for AST  
            skip_unsupported: Skip unsupported languages/pairs
        """
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.target_sr = target_sr
        self.skip_unsupported = skip_unsupported
        
        self.asr_metrics = ASRMetricsCalculator(normalizer)
        self.ast_metrics = ASTMetricsCalculator(bleu_config, chrf_config)
    
    # ========================================================================
    # ASR Evaluation
    # ========================================================================
    
    def evaluate_asr(self,
                     model: BaseASRModel,
                     dataset: BaseASRDataset,
                     language: str,
                     experiment_name: str,
                     batch_size: Optional[int] = None) -> Optional[ASREvaluationResult]:
        """Evaluate ASR model on a single language.
        
        Args:
            model: ASR model
            dataset: Dataset with monolingual audio samples
            language: FLEURS language code
            experiment_name: Name for this experiment
            batch_size: Override default batch size
            
        Returns:
            ASREvaluationResult or None if skipped
        """
        if self.skip_unsupported and not model.supports_language(language):
            print(f"⏭️  Skipping ASR {language}: not supported by model")
            return None
        
        model_name = model.get_model_info()['model_name']
        print(f"🎙️  ASR: {model_name} on {language}")
        
        samples = dataset.get_language_samples(language)
        if not samples:
            raise ValueError(f"No samples for {language}")
        
        bs = batch_size or self.batch_size
        start_time = time.time()
        predictions = self._run_asr_inference(model, samples, bs, language)
        total_time = time.time() - start_time
        
        # Calculate metrics
        hypotheses = [p.hypothesis for p in predictions]
        references = [p.reference for p in predictions]
        metrics = self.asr_metrics.calculate(hypotheses, references)
        
        result = ASREvaluationResult(
            language=language,
            model_name=model_name.replace("/", "_"),
            predictions=predictions,
            metrics=metrics,
            experiment_name=experiment_name,
            total_time=total_time,
        )
        
        self._save_asr_result(result)
        print(f"  WER: {metrics.wer:.2f}%, CER: {metrics.cer:.2f}%")
        return result
    
    def evaluate_asr_all_languages(self,
                                   model: BaseASRModel,
                                   dataset: BaseASRDataset,
                                   experiment_name: str,
                                   languages: Optional[List[str]] = None,
                                   batch_size: Optional[int] = None) -> List[ASREvaluationResult]:
        """Evaluate ASR on all (or specified) languages.
        
        Args:
            model: ASR model
            dataset: Dataset
            experiment_name: Experiment name
            languages: Specific languages to evaluate (default: all in dataset)
            batch_size: Override batch size
            
        Returns:
            List of ASREvaluationResult (excluding skipped)
        """
        langs = languages or sorted(dataset.list_languages())
        model_name = model.get_model_info()['model_name']
        print(f"Evaluating ASR: {model_name} on {len(langs)} languages")
        
        results = []
        for lang in tqdm(langs, desc="ASR languages"):
            try:
                result = self.evaluate_asr(model, dataset, lang, experiment_name, batch_size)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Error on {lang}: {e}")
                continue
        
        if results:
            self._save_asr_summary(results, experiment_name)
        return results
    
    def _run_asr_inference(self, model: BaseASRModel, samples: List[AudioSample],
                           batch_size: int, language: str) -> List[ASRPrediction]:
        """Run ASR inference on samples."""
        predictions = []
        
        for i in tqdm(range(0, len(samples), batch_size), 
                      desc="ASR inference", leave=False):
            batch = samples[i:i + batch_size]
            
            # Load audio
            audio_arrays = []
            for sample in batch:
                try:
                    audio, sr = load_audio(sample.audio_path, self.target_sr)
                    audio_arrays.append(audio)
                except Exception as e:
                    print(f"    Audio load error {sample.audio_path}: {e}")
                    audio_arrays.append(None)
            
            # Skip batch if all audio failed to load
            valid_indices = [j for j, a in enumerate(audio_arrays) if a is not None]
            if not valid_indices:
                for sample in batch:
                    predictions.append(ASRPrediction(
                        sample_id=sample.sample_id, reference=sample.transcription,
                        hypothesis="", audio_path=sample.audio_path, language=language
                    ))
                continue
            
            valid_audio = [audio_arrays[j] for j in valid_indices]
            valid_samples = [batch[j] for j in valid_indices]
            
            # Transcribe
            start = time.time()
            try:
                hypotheses = model.transcribe(valid_audio, self.target_sr, language)
                pred_time = (time.time() - start) / len(valid_audio)
            except Exception as e:
                print(f"    Transcription error: {e}")
                hypotheses = [""] * len(valid_audio)
                pred_time = None
            
            # Create predictions for valid samples
            valid_set = set(valid_indices)
            hyp_idx = 0
            for j, sample in enumerate(batch):
                if j in valid_set:
                    predictions.append(ASRPrediction(
                        sample_id=sample.sample_id,
                        reference=sample.transcription,
                        hypothesis=hypotheses[hyp_idx],
                        audio_path=sample.audio_path,
                        language=language,
                        prediction_time=pred_time,
                    ))
                    hyp_idx += 1
                else:
                    predictions.append(ASRPrediction(
                        sample_id=sample.sample_id, reference=sample.transcription,
                        hypothesis="", audio_path=sample.audio_path, language=language
                    ))
        
        return predictions
    
    # ========================================================================
    # AST Evaluation
    # ========================================================================
    
    def evaluate_ast(self,
                     model: BaseASTModel,
                     dataset: BaseASTDataset,
                     source_lang: str,
                     target_lang: str,
                     experiment_name: str,
                     batch_size: Optional[int] = None) -> Optional[ASTEvaluationResult]:
        """Evaluate AST model on a language pair.
        
        Args:
            model: AST model
            dataset: Dataset with parallel audio samples
            source_lang: Source FLEURS language code
            target_lang: Target FLEURS language code
            experiment_name: Experiment name
            batch_size: Override batch size
            
        Returns:
            ASTEvaluationResult or None if skipped
        """
        if self.skip_unsupported and not model.supports_language_pair(source_lang, target_lang):
            print(f"⏭️  Skipping AST {source_lang}→{target_lang}: not supported")
            return None
        
        model_name = model.get_model_info()['model_name']
        print(f"🌐 AST: {model_name} on {source_lang}→{target_lang}")
        
        samples = dataset.get_parallel_samples(source_lang, target_lang)
        if not samples:
            raise ValueError(f"No parallel samples for {source_lang}→{target_lang}")
        
        bs = batch_size or self.batch_size
        start_time = time.time()
        predictions = self._run_ast_inference(model, samples, bs, source_lang, target_lang)
        total_time = time.time() - start_time
        
        # Calculate metrics
        hypotheses = [p.hypothesis for p in predictions]
        references = [p.reference for p in predictions]
        metrics = self.ast_metrics.calculate(hypotheses, references)
        
        result = ASTEvaluationResult(
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model_name.replace("/", "_"),
            predictions=predictions,
            metrics=metrics,
            experiment_name=experiment_name,
            total_time=total_time,
        )
        
        self._save_ast_result(result)
        print(f"  BLEU: {metrics.bleu:.2f}, chrF++: {metrics.chrf:.2f}")
        return result
    
    def evaluate_ast_all_pairs(self,
                               model: BaseASTModel,
                               dataset: BaseASTDataset,
                               experiment_name: str,
                               pairs: Optional[List[tuple]] = None,
                               batch_size: Optional[int] = None) -> List[ASTEvaluationResult]:
        """Evaluate AST on all (or specified) language pairs."""
        if pairs is None:
            pairs = sorted(dataset.list_language_pairs())
        
        model_name = model.get_model_info()['model_name']
        print(f"Evaluating AST: {model_name} on {len(pairs)} pairs")
        
        results = []
        for src, tgt in tqdm(pairs, desc="AST pairs"):
            try:
                result = self.evaluate_ast(model, dataset, src, tgt, experiment_name, batch_size)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Error on {src}→{tgt}: {e}")
                continue
        
        if results:
            self._save_ast_summary(results, experiment_name)
        return results
    
    def _run_ast_inference(self, model: BaseASTModel, samples: List[ParallelAudioSample],
                           batch_size: int, source_lang: str, 
                           target_lang: str) -> List[ASTPrediction]:
        """Run AST inference on parallel samples."""
        predictions = []
        
        for i in tqdm(range(0, len(samples), batch_size),
                      desc="AST inference", leave=False):
            batch = samples[i:i + batch_size]
            
            # Load source audio
            audio_arrays = []
            for sample in batch:
                try:
                    audio, sr = load_audio(sample.source_audio_path, self.target_sr)
                    audio_arrays.append(audio)
                except Exception as e:
                    print(f"    Audio load error: {e}")
                    audio_arrays.append(None)
            
            valid_indices = [j for j, a in enumerate(audio_arrays) if a is not None]
            if not valid_indices:
                for sample in batch:
                    predictions.append(ASTPrediction(
                        sample_id=sample.sample_id,
                        source_transcription=sample.source_transcription,
                        reference=sample.target_transcription,
                        hypothesis="", audio_path=sample.source_audio_path,
                        source_lang=source_lang, target_lang=target_lang
                    ))
                continue
            
            valid_audio = [audio_arrays[j] for j in valid_indices]
            valid_samples = [batch[j] for j in valid_indices]
            
            start = time.time()
            try:
                hypotheses = model.translate_speech(
                    valid_audio, self.target_sr, source_lang, target_lang
                )
                pred_time = (time.time() - start) / len(valid_audio)
            except Exception as e:
                print(f"    Translation error: {e}")
                hypotheses = [""] * len(valid_audio)
                pred_time = None
            
            valid_set = set(valid_indices)
            hyp_idx = 0
            for j, sample in enumerate(batch):
                if j in valid_set:
                    predictions.append(ASTPrediction(
                        sample_id=sample.sample_id,
                        source_transcription=sample.source_transcription,
                        reference=sample.target_transcription,
                        hypothesis=hypotheses[hyp_idx],
                        audio_path=sample.source_audio_path,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        prediction_time=pred_time,
                    ))
                    hyp_idx += 1
                else:
                    predictions.append(ASTPrediction(
                        sample_id=sample.sample_id,
                        source_transcription=sample.source_transcription,
                        reference=sample.target_transcription,
                        hypothesis="", audio_path=sample.source_audio_path,
                        source_lang=source_lang, target_lang=target_lang
                    ))
        
        return predictions
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    def _save_asr_result(self, result: ASREvaluationResult):
        """Save ASR evaluation result."""
        exp_dir = self.output_dir / result.experiment_name
        
        # Save predictions
        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pred_dir / f"{result.model_name}_asr_{result.language}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "reference", "hypothesis", "audio_path"])
            for pred in result.predictions:
                writer.writerow([pred.sample_id, pred.reference, pred.hypothesis, pred.audio_path])
        
        # Save metrics
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{result.model_name}_asr_{result.language}_metrics.json"
        
        metrics_data = {
            "task": "asr",
            "model_name": result.model_name,
            "language": result.language,
            "wer": result.metrics.wer,
            "cer": result.metrics.cer,
            "num_samples": result.metrics.num_samples,
            "total_time": result.total_time,
            "avg_time_per_sample": result.total_time / len(result.predictions) if result.total_time else None,
            "metric_config": result.metrics.metric_config,
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _save_ast_result(self, result: ASTEvaluationResult):
        """Save AST evaluation result."""
        exp_dir = self.output_dir / result.experiment_name
        
        # Save predictions
        pred_dir = exp_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        csv_path = pred_dir / f"{result.model_name}_ast_{result.source_lang}_{result.target_lang}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "source_transcription", "reference", "hypothesis", "audio_path"
            ])
            for pred in result.predictions:
                writer.writerow([
                    pred.sample_id, pred.source_transcription, pred.reference,
                    pred.hypothesis, pred.audio_path
                ])