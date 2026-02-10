# stt_benchmark/datasets/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AudioSample:
    """A single audio sample with transcription and metadata."""
    audio_path: str
    transcription: str
    language: str                          # FLEURS code (e.g., 'sw_ke')
    sample_id: str


@dataclass
class ParallelAudioSample:
    """A parallel sample for AST evaluation: source audio + target text."""
    source_audio_path: str
    source_transcription: str
    source_language: str                   # FLEURS code
    target_transcription: str
    target_language: str                   # FLEURS code
    sample_id: str


class BaseASRDataset(ABC):
    """Abstract base class for ASR datasets (monolingual audio + transcription)."""
    
    @abstractmethod
    def get_language_samples(self, language: str) -> List[AudioSample]:
        """Get all samples for a language."""
        pass
    
    @abstractmethod
    def list_languages(self) -> Set[str]:
        """List available language codes."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        pass
    
    def has_language(self, language: str) -> bool:
        """Check if a language is available."""
        return language in self.list_languages()
    
    def get_language_batch(self, language: str, 
                           batch_size: int = 1) -> Iterator[List[AudioSample]]:
        """Get samples in batches."""
        samples = self.get_language_samples(language)
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]


class BaseASTDataset(ABC):
    """Abstract base class for AST datasets (parallel audio + cross-lingual text)."""
    
    @abstractmethod
    def get_parallel_samples(self, source_lang: str, 
                             target_lang: str) -> List[ParallelAudioSample]:
        """Get parallel samples for a language pair."""
        pass
    
    @abstractmethod
    def list_language_pairs(self) -> Set[tuple]:
        """List available (source, target) language pairs."""
        pass
    
    def has_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if a language pair is available."""
        return (source_lang, target_lang) in self.list_language_pairs()
    
    def get_parallel_batch(self, source_lang: str, target_lang: str,
                           batch_size: int = 1) -> Iterator[List[ParallelAudioSample]]:
        """Get parallel samples in batches."""
        samples = self.get_parallel_samples(source_lang, target_lang)
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]