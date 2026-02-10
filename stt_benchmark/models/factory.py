# stt_benchmark/models/factory.py

import yaml
from typing import Dict, Any
from pathlib import Path
from stt_benchmark.models.hf.whisper import WhisperModel
from stt_benchmark.models.hf.mms import MMSModel
from stt_benchmark.models.hf.seamless import SeamlessModel

class ModelFactory:
    """Factory for creating STT models."""