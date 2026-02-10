# STT Benchmark

A benchmarking framework for evaluating speech-to-text models on the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset, with a focus on African languages. Supports Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST) evaluation across Whisper, SeamlessM4T, and MMS models.

## Setup

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# Answer yes to terms and to automatically setting up Miniconda
# Reopen terminal

# Create environment
conda deactivate
conda create -n stt python=3.10
conda activate stt

# Install the package
pip install -e .
```

## Evaluation

### Using a config file (recommended)

Define an evaluation config in YAML:

```yaml
# configs/african_evaluation.yaml
experiment_name: african_eval

asr:
  languages:
    - sw_ke    # Swahili
    - yo_ng    # Yoruba
    - zu_za    # Zulu
    - am_et    # Amharic

ast:
  anchors:
    - en_us    # English
    - fr_fr    # French
  direction: both   # source → anchor AND anchor → source
```

Run the evaluation:

```bash
python scripts/evaluate.py whisper_large_v3 \
    --eval-config configs/african_evaluation.yaml \
    --dataset-path /path/to/FLEURS/splits/test
```

### Ad-hoc evaluation

```bash
# ASR on a single language
python scripts/evaluate.py whisper_large_v3 --task asr --language sw_ke

# AST on a single pair
python scripts/evaluate.py seamless_m4t_v2_large --task ast --source-lang sw_ke --target-lang en_us
```

### Available models

| Model ID                  | Type     | Tasks    |
|---------------------------|----------|----------|
| `whisper_large_v3`        | Whisper  | ASR, AST |
| `whisper_large_v3_turbo`  | Whisper  | ASR, AST |
| `whisper_medium`          | Whisper  | ASR, AST |
| `whisper_small`           | Whisper  | ASR, AST |
| `seamless_m4t_v2_large`   | Seamless | ASR, AST |
| `mms_1b_all`              | MMS      | ASR      |
| `mms_1b_fl102`            | MMS      | ASR      |

> **Note:** Whisper AST only translates to English. SeamlessM4T supports many language pairs.

## Summarizing results

After running evaluations, generate summary CSVs and a terminal report:

```bash
# Summarize a single experiment
python scripts/summarize_results.py results/african_eval

# Summarize all experiments
python scripts/summarize_results.py results/

# Write CSVs to a custom directory
python scripts/summarize_results.py results/ -o reports/
```

This produces `asr_summary.csv` and `ast_summary.csv` with per-model, per-language metrics (WER/CER for ASR, BLEU/chrF++ for AST).