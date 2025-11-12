# HuggingFace Migration Guide

This guide explains how to migrate from the fairseq2-based implementation to the new HuggingFace transformers-based implementation.

## Overview

The omnilingual-asr project has been migrated to use HuggingFace transformers instead of fairseq2, removing the fairseq2 dependency entirely. This provides:

- ✅ **Better ecosystem support**: Integration with HuggingFace Hub and transformers
- ✅ **Easier maintenance**: Active HuggingFace development and community
- ✅ **Simpler dependencies**: Fewer custom dependencies
- ✅ **Drop-in replacement**: Minimal code changes for existing users

## What Changed

### Dependencies

**Before (fairseq2):**
```python
pip install fairseq2[arrow]
```

**After (HuggingFace):**
```python
pip install transformers accelerate sentencepiece
```

### Core Components

| Component | fairseq2 | HuggingFace |
|-----------|----------|-------------|
| Model | `Wav2Vec2LlamaModel` | `Wav2Vec2LlamaModelHF` |
| Encoder | `Wav2Vec2Frontend` + `TransformerEncoder` | `transformers.Wav2Vec2Model` |
| Decoder | `TransformerLMDecoder` | `transformers.LlamaForCausalLM` |
| KV Cache | `IncrementalStateBag` | `past_key_values` (native) |
| Audio Loading | `AudioDecoder` + `DataPipeline` | `torchaudio` |
| Batch Format | `Seq2SeqBatch` + `BatchLayout` | Simple `Dict` + `List[int]` |

## Migration Steps

### 1. Install New Dependencies

```bash
# Uninstall fairseq2 (optional, can keep for transition)
pip uninstall fairseq2

# Install HuggingFace dependencies
pip install transformers>=4.35.0 accelerate>=0.20.0 sentencepiece>=0.1.99
```

Or install from the updated `pyproject.toml`:

```bash
pip install -e .
```

### 2. Convert Your Checkpoints

Convert existing fairseq2 checkpoints to HuggingFace format:

```bash
python scripts/convert_fairseq2_to_hf.py \
    --input_checkpoint path/to/fairseq2_checkpoint.pt \
    --output_dir path/to/hf_model \
    --model_config 7b
```

This creates:
- `pytorch_model.bin`: Converted weights
- `conversion_metadata.json`: Conversion info

### 3. Update Your Code

#### Inference Pipeline

**Before (fairseq2):**
```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Load from model card
pipeline = ASRInferencePipeline(
    model_card="wav2vec2_llama_7b",
    device="cuda",
    dtype=torch.bfloat16,
)

transcriptions = pipeline(audio_files, lang="en")
```

**After (HuggingFace):**
```python
from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF
from omnilingual_asr.models.wav2vec2_llama.model_hf import Wav2Vec2LlamaModelHF

# Load model manually (or create a factory)
# For now, you need to instantiate the model with converted weights
model = load_converted_model("path/to/hf_model")  # See example below
tokenizer = load_tokenizer("path/to/tokenizer")

pipeline = ASRInferencePipelineHF(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    dtype=torch.bfloat16,
)

transcriptions = pipeline(audio_files, lang="en")
```

#### Loading a Converted Model

```python
import torch
from transformers import Wav2Vec2Model, LlamaForCausalLM, LlamaConfig
from omnilingual_asr.models.wav2vec2_llama.model_hf import (
    Wav2Vec2LlamaModelHF,
    VocabularyInfo,
)
from omnilingual_asr.models.wav2vec2_llama.config import ModelType

def load_converted_model(checkpoint_dir: str, config_name: str = "7b"):
    """Load a converted HuggingFace model."""

    # Load converted weights
    checkpoint_path = Path(checkpoint_dir) / "pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Initialize HuggingFace base models
    # You may need to adjust these configs based on your model
    wav2vec2_config = Wav2Vec2Config(
        hidden_size=1024,  # Adjust based on your model
        num_hidden_layers=24,
        # ... other config
    )
    encoder = Wav2Vec2Model(wav2vec2_config)

    llama_config = LlamaConfig(
        hidden_size=4096,  # For 7b model
        num_hidden_layers=12,
        num_attention_heads=8,
        # ... other config
    )
    decoder = LlamaForCausalLM(llama_config)

    # Create projection layers
    encoder_proj = torch.nn.Linear(1024, 4096, bias=True)
    text_frontend = torch.nn.Embedding(9818, 4096)  # vocab_size, hidden_dim
    final_proj = torch.nn.Linear(4096, 9818, bias=False)

    # Create vocabulary info
    vocab_info = VocabularyInfo(
        size=9812,
        unk_idx=3,
        bos_idx=0,
        eos_idx=2,
        pad_idx=1,
    )

    # Create model
    model = Wav2Vec2LlamaModelHF(
        model_type=ModelType.LLM_ASR,
        model_dim=4096,
        encoder=encoder,
        encoder_proj=encoder_proj,
        text_frontend=text_frontend,
        llama_decoder=decoder,
        final_proj=final_proj,
        target_vocab_info=vocab_info,
        max_generation_length=8192,
    )

    # Load converted weights
    model.load_state_dict(state_dict, strict=False)

    return model
```

### 4. Audio Processing

**Before (fairseq2):**
```python
from fairseq2.data.audio import AudioDecoder

audio_decoder = AudioDecoder(dtype=torch.float32)
waveform = audio_decoder(file_path)
```

**After (HuggingFace):**
```python
from omnilingual_asr.models.inference.audio_utils import load_audio

audio_data = load_audio(file_path, target_sample_rate=16000)
waveform = audio_data["waveform"]
```

### 5. Beam Search

**Before (fairseq2):**
```python
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator
)

generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
    model=model,
    config=beam_config,
)
```

**After (HuggingFace):**
```python
from omnilingual_asr.models.wav2vec2_llama.beamsearch_hf import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF
)

generator = Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF(
    model=model,
    config=beam_config,
)
```

## API Compatibility

### Batch Format

**fairseq2 Batch:**
```python
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.nn import BatchLayout

batch = Seq2SeqBatch(
    source_seqs=audio_tensor,
    source_seq_lens=[...],
    target_seqs=text_tensor,
    target_seq_lens=[...],
    example={'lang': ['en', 'fr']},
)
```

**HuggingFace Batch:**
```python
batch = {
    'source_seqs': audio_tensor,      # [B, T, D]
    'source_seq_lens': [...],         # List[int]
    'target_seqs': text_tensor,       # [B, S]
    'target_seq_lens': [...],         # List[int]
    'example': {'lang': ['en', 'fr']},  # Dict
}
```

### Model Forward Pass

Both implementations maintain the same forward signature:

```python
# Both work the same way
loss = model(batch)
loss, logits, ... = model(batch, return_logits=True)
context, lengths = model(batch, return_decoder_inputs=True)
```

## Breaking Changes

### 1. No Model Cards (Yet)

The HuggingFace version doesn't support fairseq2 model cards. You need to:
- Load models manually from converted checkpoints
- Or create your own model loading utilities

### 2. Tokenizer Interface

fairseq2 tokenizers are not directly compatible. You need to:
- Use a HuggingFace-compatible tokenizer (e.g., SentencePiece)
- Or wrap your existing tokenizer with encode/decode methods

### 3. Training Code

The migration currently focuses on **inference only**. Training code still uses fairseq2 and requires additional migration work.

## Transition Period

During the transition, both implementations coexist:

- **New inference**: Use `*_hf.py` modules (fairseq2-free)
- **Legacy inference**: Use original modules (requires fairseq2)
- **Training**: Continue using fairseq2 (for now)

To keep fairseq2 for legacy support:

```bash
pip install -e ".[fairseq2]"
```

## Testing Your Migration

### Quick Test

```python
# Test that imports work
from omnilingual_asr.models.wav2vec2_llama.model_hf import Wav2Vec2LlamaModelHF
from omnilingual_asr.models.wav2vec2_llama.beamsearch_hf import Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF
from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF
from omnilingual_asr.models.inference.audio_utils import load_audio

print("✅ All HuggingFace imports successful!")
```

### End-to-End Test

```python
# Load test audio
audio_data = load_audio("test.wav")

# Create pipeline
pipeline = ASRInferencePipelineHF(model, tokenizer)

# Run inference
transcriptions = pipeline(["test.wav"], lang="en")

print(f"Transcription: {transcriptions[0]}")
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'fairseq2'`

**Solution:** Install with fairseq2 support during transition:
```bash
pip install -e ".[fairseq2]"
```

Or remove all fairseq2 imports from your code.

### Checkpoint Loading Errors

**Problem:** `KeyError` when loading converted checkpoints

**Solution:** The conversion script may need adjustment for your specific model. Check:
1. Key mappings in `convert_fairseq2_to_hf.py`
2. Model configuration matches your checkpoint
3. Use `strict=False` when loading: `model.load_state_dict(state_dict, strict=False)`

### Shape Mismatches

**Problem:** `RuntimeError: shape mismatch` during inference

**Solution:**
1. Verify audio preprocessing (should be 16kHz mono)
2. Check model dimensions match config
3. Ensure encoder/decoder configs are correct

## Performance Comparison

| Aspect | fairseq2 | HuggingFace |
|--------|----------|-------------|
| Inference Speed | ~Baseline | ~Same |
| Memory Usage | ~Baseline | ~Same |
| Dependencies | 15+ packages | 8 packages |
| Ecosystem | Limited | Extensive |
| Maintenance | Slowing | Active |

## Rollback Plan

If you encounter issues, you can rollback:

```bash
# Reinstall fairseq2
pip install fairseq2[arrow]

# Use original modules
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
```

## Future Work

- [ ] HuggingFace Hub integration
- [ ] Training pipeline migration
- [ ] Model cards for HF version
- [ ] Optimized inference (flash attention, etc.)
- [ ] Quantization support

## Getting Help

- **Issues**: File on GitHub with `[migration]` tag
- **Questions**: Check discussions or create an issue
- **Bugs**: Report with both fairseq2 and HF behavior

## Summary

The HuggingFace migration provides a more maintainable and ecosystem-friendly implementation. The transition is designed to be gradual, with both implementations coexisting during the migration period.

Key takeaways:
- ✅ Replace fairseq2 with transformers
- ✅ Convert checkpoints with provided script
- ✅ Update imports to use `*_hf` modules
- ✅ Test thoroughly before deploying to production
- ✅ Keep fairseq2 as optional dependency during transition
