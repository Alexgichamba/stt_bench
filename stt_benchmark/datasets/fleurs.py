# stt_benchmark/datasets/fleurs.py

"""
FLEURS dataset loader for preprocessed CSV files.

Expected directory structure:
    FLEURS/splits/{split}/
        sw_ke.csv                    # Monolingual: id, path, transcript, gender
        sw_ke-en_us.csv              # Parallel: sw_ke-id, sw_ke-path, sw_ke-transcript,
                                     #           sw_ke-gender, en_us-id, en_us-path, ...
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from tqdm import tqdm
from stt_benchmark.datasets.base import (
    BaseASRDataset, BaseASTDataset, AudioSample, ParallelAudioSample,
)


class FleursDataset(BaseASRDataset, BaseASTDataset):
    """FLEURS dataset loader for preprocessed CSVs.

    Supports both ASR (monolingual) and AST (parallel) evaluation.
    """

    def __init__(self, dataset_path: str, lazy_load: bool = True):
        """Initialize FLEURS dataset.

        Args:
            dataset_path: Path to split directory (e.g., 'FLEURS/splits/test')
            lazy_load: If True, only discover files on init; load data on demand.
                       If False, load all CSVs into memory immediately.
        """
        self.dataset_path = Path(dataset_path)
        self.lazy_load = lazy_load

        # Caches
        self._mono_cache: Dict[str, List[AudioSample]] = {}
        self._parallel_cache: Dict[str, List[ParallelAudioSample]] = {}

        # Discover available files
        self._mono_files: Dict[str, Path] = {}      # lang -> csv path
        self._parallel_files: Dict[str, Path] = {}   # "src-tgt" -> csv path
        self._discover_files()

        if not lazy_load:
            self._load_all()

    def _discover_files(self):
        """Discover available CSV files."""
        for csv_file in self.dataset_path.glob("*.csv"):
            stem = csv_file.stem
            if "-" in stem:
                self._parallel_files[stem] = csv_file
            else:
                self._mono_files[stem] = csv_file

        print(f"Discovered {len(self._mono_files)} monolingual + "
              f"{len(self._parallel_files)} parallel files in {self.dataset_path}")

    def _load_all(self):
        """Load all data into memory."""
        for lang in tqdm(self._mono_files, desc="Loading monolingual"):
            self._load_mono(lang)
        for pair_key in tqdm(self._parallel_files, desc="Loading parallel"):
            src, tgt = pair_key.split("-", 1)
            self._load_parallel(src, tgt)

    def _load_mono(self, language: str) -> List[AudioSample]:
        """Load monolingual CSV for a language."""
        if language in self._mono_cache:
            return self._mono_cache[language]

        csv_path = self._mono_files.get(language)
        if not csv_path:
            raise ValueError(f"No monolingual data for '{language}'. "
                           f"Available: {sorted(self._mono_files.keys())}")

        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = AudioSample(
                    audio_path=row['path'],
                    transcription=row['transcript'],
                    language=language,
                    sample_id=row['id'],
                )
                samples.append(sample)

        self._mono_cache[language] = samples
        return samples

    def _load_parallel(self, source_lang: str, target_lang: str) -> List[ParallelAudioSample]:
        """Load parallel CSV for a language pair."""
        pair_key = f"{source_lang}-{target_lang}"
        if pair_key in self._parallel_cache:
            return self._parallel_cache[pair_key]

        csv_path = self._parallel_files.get(pair_key)
        if not csv_path:
            raise ValueError(f"No parallel data for '{pair_key}'. "
                           f"Available: {sorted(self._parallel_files.keys())}")

        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Column names use the FLEURS code as prefix:
                #   sw_ke-id, sw_ke-path, sw_ke-transcript, sw_ke-gender,
                #   en_us-id, en_us-path, en_us-transcript, en_us-gender
                sample = ParallelAudioSample(
                    source_audio_path=row[f'{source_lang}-path'],
                    source_transcription=row[f'{source_lang}-transcript'],
                    source_language=source_lang,
                    target_transcription=row[f'{target_lang}-transcript'],
                    target_language=target_lang,
                    sample_id=row[f'{source_lang}-id'],
                )
                samples.append(sample)

        self._parallel_cache[pair_key] = samples
        return samples

    # ---- BaseASRDataset interface ----

    def get_language_samples(self, language: str) -> List[AudioSample]:
        return self._load_mono(language)

    def list_languages(self) -> Set[str]:
        return set(self._mono_files.keys())

    def get_dataset_info(self) -> Dict[str, Any]:
        mono_counts = {}
        for lang in self._mono_files:
            if lang in self._mono_cache:
                mono_counts[lang] = len(self._mono_cache[lang])

        return {
            "dataset_name": "FLEURS",
            "dataset_path": str(self.dataset_path),
            "monolingual_languages": sorted(self._mono_files.keys()),
            "num_monolingual_languages": len(self._mono_files),
            "parallel_pairs": sorted(self._parallel_files.keys()),
            "num_parallel_pairs": len(self._parallel_files),
            "cached_mono_counts": mono_counts,
        }

    # ---- BaseASTDataset interface ----

    def get_parallel_samples(self, source_lang: str,
                             target_lang: str) -> List[ParallelAudioSample]:
        return self._load_parallel(source_lang, target_lang)

    def list_language_pairs(self) -> Set[tuple]:
        pairs = set()
        for pair_key in self._parallel_files:
            src, tgt = pair_key.split("-", 1)
            pairs.add((src, tgt))
        return pairs

    def get_languages(self) -> Set[str]:
        """Get all language codes (mono + parallel sources)."""
        langs = set(self._mono_files.keys())
        for pair_key in self._parallel_files:
            src, tgt = pair_key.split("-", 1)
            langs.add(src)
            langs.add(tgt)
        return langs