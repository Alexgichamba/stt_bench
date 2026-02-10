# stt_benchmark/utils/audio.py

"""Audio loading and preprocessing utilities."""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sampling rate.
    
    Args:
        path: Path to audio file (wav, flac, mp3, etc.)
        target_sr: Target sampling rate (default 16000 for most STT models)
        
    Returns:
        Tuple of (audio_array, sampling_rate)
    """
    audio, sr = sf.read(path, dtype='float32')
    
    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def get_audio_duration(audio: np.ndarray, sampling_rate: int) -> float:
    """Get audio duration in seconds."""
    return len(audio) / sampling_rate


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad (with zeros) or trim audio to target length in samples."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio