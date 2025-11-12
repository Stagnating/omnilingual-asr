# Scripts Directory

This directory contains utility scripts and examples for omnilingual-asr.

## HuggingFace Migration Scripts

### Checkpoint Conversion

**`convert_fairseq2_to_hf.py`** - Convert fairseq2 checkpoints to HuggingFace format

```bash
python scripts/convert_fairseq2_to_hf.py \
    --input_checkpoint path/to/fairseq2_checkpoint.pt \
    --output_dir path/to/output \
    --model_config 300m
```

**Supported model configs:** `300m`, `1b`, `3b`, `7b`, `7b_zs`

**Output:**
- `pytorch_model.bin` - Converted weights
- `conversion_metadata.json` - Conversion info

## HuggingFace Inference Examples

### Complete Example (300m)

**`example_hf_inference_300m.py`** - Full-featured inference example

```bash
# Basic usage
python scripts/example_hf_inference_300m.py \
    --checkpoint_dir path/to/converted_checkpoint \
    --audio_files audio1.wav audio2.wav \
    --lang en

# Advanced usage
python scripts/example_hf_inference_300m.py \
    --checkpoint_dir converted_300m \
    --audio_files audio1.wav audio2.wav \
    --lang en \
    --device cuda \
    --dtype bfloat16 \
    --beam_size 10
```

**Features:**
- ✅ Complete 300m model loading
- ✅ Checkpoint weight loading
- ✅ Configurable beam search
- ✅ Multi-file batch processing
- ✅ Language hints
- ✅ Device/dtype selection
- ✅ Detailed logging

**Options:**
```
--checkpoint_dir    Directory with pytorch_model.bin (required)
--audio_files       Audio files to transcribe (required)
--tokenizer_path    Path to tokenizer (optional)
--lang              Language code (optional)
--device            Device: cuda/cpu (default: auto)
--dtype             Data type: float32/float16/bfloat16 (default: bfloat16)
--beam_size         Beam size for decoding (default: 5)
```

### Quick Start (Minimal)

**`quickstart_hf_300m.py`** - Minimal working example

```bash
python scripts/quickstart_hf_300m.py audio.wav
```

**Features:**
- ✅ Minimal code (~70 lines)
- ✅ Shows core architecture
- ✅ Works without checkpoint (random weights)
- ✅ CPU inference for testing

## Workflow

### 1. Convert Checkpoint

```bash
# Convert your fairseq2 checkpoint
python scripts/convert_fairseq2_to_hf.py \
    --input_checkpoint my_300m_model.pt \
    --output_dir converted_300m \
    --model_config 300m
```

### 2. Run Inference

```bash
# Use the full example
python scripts/example_hf_inference_300m.py \
    --checkpoint_dir converted_300m \
    --audio_files test.wav \
    --lang en \
    --device cuda
```

### 3. Integrate into Your Code

Use the examples as reference to integrate into your own code:

```python
from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF

# Load model (see example_hf_inference_300m.py for details)
model = load_300m_model("converted_300m/pytorch_model.bin")
tokenizer = load_tokenizer("path/to/tokenizer")

# Create pipeline
pipeline = ASRInferencePipelineHF(model, tokenizer, device="cuda")

# Transcribe
transcriptions = pipeline(["audio.wav"], lang="en")
print(transcriptions[0])
```

## Model Sizes

| Model | Parameters | Hidden Size | Layers | Config Name |
|-------|-----------|-------------|--------|-------------|
| 300m  | ~300M     | 4096        | 12     | `300m`      |
| 1b    | ~1B       | 4096        | 12     | `1b`        |
| 3b    | ~3B       | 4096        | 12     | `3b`        |
| 7b    | ~7B       | 4096        | 12     | `7b`        |

All models use:
- Encoder: Wav2Vec2 (1024 dim, 24 layers)
- Decoder: LLaMA (4096 dim, 12 layers)
- Vocab: ~9.8k tokens

## Requirements

For HuggingFace examples:

```bash
pip install transformers>=4.35.0 accelerate>=0.20.0 torch torchaudio sentencepiece
```

Or install from the project:

```bash
pip install -e .
```

## Troubleshooting

### "Checkpoint not found"

Make sure you've converted your checkpoint first:
```bash
python scripts/convert_fairseq2_to_hf.py --input_checkpoint ... --output_dir ...
```

### "Out of memory"

Try:
- Smaller beam size: `--beam_size 1`
- Lower precision: `--dtype float16`
- CPU inference: `--device cpu`

### "Missing keys in checkpoint"

This is normal if using `strict=False`. The model will initialize missing weights randomly. For production, ensure all keys are present.

### "Module not found"

Make sure you've installed the package:
```bash
pip install -e .
```

## Getting Help

- See [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) for detailed migration instructions
- See [HUGGINGFACE_INFERENCE.md](../HUGGINGFACE_INFERENCE.md) for comprehensive usage guide
- File issues on GitHub with `[huggingface]` tag

## What's Next?

After running the examples:

1. **Production use**: Load your own converted checkpoints
2. **Custom tokenizer**: Replace `SimpleTokenizer` with your real tokenizer
3. **Batch processing**: Process multiple files efficiently
4. **Integration**: Use the pipeline in your application

See the documentation files for more advanced usage patterns!
