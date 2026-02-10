# stt_benchmark/models/hf/whisper.py

"""Hugging Face implementation of OpenAI Whisper STT models."""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from stt_benchmark.models.base import BaseSTTModel

class WhisperModel(BaseSTTModel):
    """Hugging Face implementation of OpenAI Whisper models."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name=model_name, device=device)