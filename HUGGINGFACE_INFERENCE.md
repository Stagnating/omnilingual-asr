# HuggingFace Inference Options

This document describes the new HuggingFace-based inference implementation for omnilingual-asr that **does not require fairseq2**.

## Quick Start

### Installation

```bash
# Install HuggingFace dependencies (no fairseq2 needed!)
pip install transformers>=4.35.0 accelerate>=0.20.0 sentencepiece>=0.1.99 torch torchaudio
```

### Basic Usage

```python
from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF

# Initialize pipeline (you need to provide model and tokenizer)
pipeline = ASRInferencePipelineHF(
    model=your_model,  # Wav2Vec2LlamaModelHF instance
    tokenizer=your_tokenizer,
    device="cuda",
    dtype=torch.bfloat16,
)

# Transcribe audio files
transcriptions = pipeline(
    audio=["audio1.wav", "audio2.mp3"],
    lang="en",  # Optional language hint
)

print(transcriptions[0])  # First transcription
```

## Architecture Overview

The HuggingFace implementation consists of:

### 1. Model Components

```
┌─────────────────────────────────────────────────────────┐
│                  Wav2Vec2LlamaModelHF                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐         ┌────────────────────┐   │
│  │  Wav2Vec2Model  │────────▶│  Encoder Projection│   │
│  │  (HF)           │         │  Linear Layer      │   │
│  └─────────────────┘         └────────────────────┘   │
│         │                              │               │
│         │ Audio Embeddings             │               │
│         └──────────────────────────────┘               │
│                     ▼                                   │
│         ┌──────────────────────────┐                   │
│         │  Concat with Text Embeds │                   │
│         └──────────────────────────┘                   │
│                     ▼                                   │
│         ┌──────────────────────────┐                   │
│         │  LlamaForCausalLM  (HF)  │                   │
│         │  with past_key_values    │                   │
│         └──────────────────────────┘                   │
│                     ▼                                   │
│         ┌──────────────────────────┐                   │
│         │    Final Projection      │                   │
│         │    to Vocabulary         │                   │
│         └──────────────────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2. Beam Search

```python
from omnilingual_asr.models.wav2vec2_llama.beamsearch_hf import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF
)

# Beam search configuration
from omnilingual_asr.models.wav2vec2_llama.config import (
    Wav2Vec2LlamaBeamSearchConfig
)

beam_config = Wav2Vec2LlamaBeamSearchConfig(
    nbest=5,              # Beam size
    length_norm=False,    # Length normalization
    compression_window=100,    # Early stopping window
    compression_threshold=4.0,  # Compression ratio threshold
)

generator = Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF(
    model=model,
    config=beam_config,
)

# Use in inference
hypotheses, lengths = generator.generate_hypotheses(
    decoder_context_inputs=context_embeddings,
    decoder_context_lengths=context_lengths,
)
```

## Key Features

### 1. No fairseq2 Dependency

The implementation uses only standard PyTorch and HuggingFace libraries:

- ✅ `transformers` for model architectures
- ✅ `torch` for tensor operations
- ✅ `torchaudio` for audio loading
- ❌ No `fairseq2` required!

### 2. Native HuggingFace Integration

- **Model Architecture**: Uses `transformers.Wav2Vec2Model` and `transformers.LlamaForCausalLM`
- **KV Caching**: Native `past_key_values` (no custom IncrementalStateBag)
- **Beam Search**: Custom implementation using HF's cache mechanism

### 3. Flexible Audio Input

```python
# Supports multiple input formats
audio_inputs = [
    "path/to/audio.wav",           # File path (str)
    Path("path/to/audio.mp3"),     # File path (Path)
    audio_bytes,                    # Raw bytes
    numpy_array,                    # NumPy array
    {"waveform": tensor, "sample_rate": 16000},  # Pre-loaded dict
]

transcriptions = pipeline(audio_inputs)
```

### 4. Automatic Audio Processing

The pipeline handles:
- Multiple audio formats (WAV, MP3, FLAC, etc.)
- Automatic resampling to 16kHz
- Mono conversion (if stereo)
- Length validation (max 40 seconds by default)

## Model Loading

### From Converted Checkpoint

```python
import torch
from transformers import Wav2Vec2Config, LlamaConfig, Wav2Vec2Model, LlamaForCausalLM
from omnilingual_asr.models.wav2vec2_llama.model_hf import (
    Wav2Vec2LlamaModelHF,
    VocabularyInfo,
)
from omnilingual_asr.models.wav2vec2_llama.config import ModelType

# Step 1: Define model configuration
encoder_config = Wav2Vec2Config(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    # ... other Wav2Vec2 config
)

decoder_config = LlamaConfig(
    hidden_size=4096,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=8,
    intermediate_size=4096,
    vocab_size=9818,
    # ... other LLaMA config
)

# Step 2: Initialize base models
encoder = Wav2Vec2Model(encoder_config)
decoder = LlamaForCausalLM(decoder_config)

# Step 3: Create projection and embedding layers
encoder_proj = torch.nn.Linear(1024, 4096, bias=True)
text_frontend = torch.nn.Embedding(9818, 4096)
final_proj = torch.nn.Linear(4096, 9812, bias=False)

# Step 4: Create vocabulary info
vocab_info = VocabularyInfo(
    size=9812,
    unk_idx=3,
    bos_idx=0,
    eos_idx=2,
    pad_idx=1,
)

# Step 5: Assemble the model
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
    encoder_stacking=1,
)

# Step 6: Load converted weights
checkpoint = torch.load("path/to/pytorch_model.bin")
model.load_state_dict(checkpoint, strict=False)

# Step 7: Move to device
model = model.to("cuda", dtype=torch.bfloat16)
model.eval()
```

## Audio Processing Utilities

### Load Audio

```python
from omnilingual_asr.models.inference.audio_utils import load_audio

# Load from file
audio_data = load_audio("audio.wav", target_sample_rate=16000)
# Returns: {"waveform": Tensor, "sample_rate": int}

# Load from bytes
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()
audio_data = load_audio(audio_bytes, target_sample_rate=16000)

# Load from numpy
import numpy as np
waveform_np = np.random.randn(16000)  # 1 second at 16kHz
audio_data = load_audio(waveform_np, target_sample_rate=16000)
```

### Resample Audio

```python
from omnilingual_asr.models.inference.audio_utils import resample_audio

audio_data = {
    "waveform": torch.randn(1, 48000),  # 48kHz audio
    "sample_rate": 48000,
}

resampled = resample_audio(audio_data, target_sample_rate=16000)
# Now: {"waveform": Tensor[1, 16000], "sample_rate": 16000}
```

### Validate Length

```python
from omnilingual_asr.models.inference.audio_utils import validate_audio_length

# Raises error if audio > 40 seconds
audio_data = validate_audio_length(
    audio_data,
    max_seconds=40.0,
    sample_rate=16000,
)
```

## Batch Processing

```python
# Process multiple files efficiently
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
languages = ["en", "es", "fr"]

# Batch inference
transcriptions = pipeline(
    audio=audio_files,
    lang=languages,  # Language hints (optional)
)

for file, trans in zip(audio_files, transcriptions):
    print(f"{file}: {trans}")
```

## Advanced Configuration

### Custom Beam Search

```python
from omnilingual_asr.models.wav2vec2_llama.config import (
    Wav2Vec2LlamaBeamSearchConfig
)

# Customize beam search parameters
beam_config = Wav2Vec2LlamaBeamSearchConfig(
    nbest=10,                      # Larger beam for better quality
    length_norm=True,              # Enable length normalization
    compression_window=50,         # Smaller window for faster stopping
    compression_threshold=3.5,     # Lower threshold for earlier stopping
)

pipeline = ASRInferencePipelineHF(
    model=model,
    tokenizer=tokenizer,
    beam_search_config=beam_config,
)
```

### Memory Optimization

```python
# Use float16 for lower memory usage
pipeline = ASRInferencePipelineHF(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    dtype=torch.float16,  # Or torch.bfloat16
)

# Clear cache after inference
import torch
transcriptions = pipeline(audio)
torch.cuda.empty_cache()
```

## Model Types

The implementation supports three model types:

### 1. LLM-ASR (Standard)

```python
model_type = ModelType.LLM_ASR

# Syntax: audio <bos> text <eos>
```

### 2. LLM-ASR-LID (Language ID)

```python
model_type = ModelType.LLM_ASR_LID
lang_embeddings_p = 0.5  # Probability of using language embeddings

# Requires language embeddings and mapping
lang_embeddings = torch.nn.Embedding(num_languages, hidden_dim)
lang_mapping = {"en": 1, "es": 2, "fr": 3, ...}

# Syntax: audio <special> lang <bos> text <eos>
```

### 3. Zero-Shot

```python
model_type = ModelType.ZERO_SHOT
n_context_examples = 10

# Requires context examples
# Syntax: <ctx> (<ex> ctx_audio <bos> ctx_text <eos> </ex>)* </ctx> audio <bos> text <eos>
```

## Checkpoint Conversion

Convert existing fairseq2 checkpoints:

```bash
python scripts/convert_fairseq2_to_hf.py \
    --input_checkpoint path/to/fairseq2_checkpoint.pt \
    --output_dir path/to/output \
    --model_config 7b
```

This creates:
- `pytorch_model.bin`: Converted model weights
- `conversion_metadata.json`: Conversion information

## Performance Tips

1. **Use bfloat16**: Better numerical stability than float16
   ```python
   dtype=torch.bfloat16
   ```

2. **Batch similar lengths**: Group audio files of similar duration
   ```python
   # Sort by duration before batching
   audio_files.sort(key=lambda x: get_duration(x))
   ```

3. **GPU memory**: Monitor and clear cache
   ```python
   torch.cuda.empty_cache()
   ```

4. **Beam size**: Smaller beam = faster inference
   ```python
   beam_config = Wav2Vec2LlamaBeamSearchConfig(nbest=1)  # Greedy
   ```

## Comparison with fairseq2 Version

| Feature | fairseq2 | HuggingFace |
|---------|----------|-------------|
| Dependencies | fairseq2, ~15 packages | transformers, ~8 packages |
| Model Loading | Model cards | Manual construction |
| Audio Processing | AudioDecoder | torchaudio |
| Batch Format | Seq2SeqBatch | Dict |
| KV Cache | IncrementalStateBag | past_key_values |
| Ecosystem | Limited | Extensive |
| Inference Speed | Baseline | ~Same |
| Memory Usage | Baseline | ~Same |

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
pipeline(audio[:1])  # Process one at a time

# Use smaller beam
beam_config = Wav2Vec2LlamaBeamSearchConfig(nbest=1)

# Use float16
dtype=torch.float16
```

### Slow Inference

```python
# Use greedy decoding (nbest=1)
# Enable CUDA graphs (if supported)
# Batch similar-length audio together
```

### Quality Issues

```python
# Increase beam size
beam_config = Wav2Vec2LlamaBeamSearchConfig(nbest=10)

# Enable length normalization
beam_config.length_norm = True

# Check audio quality (16kHz, mono)
```

## Examples

See the [examples](examples/) directory for complete working examples:
- `examples/hf_basic_inference.py`: Basic transcription
- `examples/hf_batch_inference.py`: Batch processing
- `examples/hf_custom_model.py`: Custom model loading

## Next Steps

- **Training**: Training migration coming soon
- **HF Hub**: Model card support in development
- **Optimizations**: Flash attention, quantization planned

## Support

For issues or questions:
- Check the [Migration Guide](MIGRATION_GUIDE.md)
- File an issue with `[huggingface]` tag
- See discussions for community help
