#!/usr/bin/env python3
"""
Quick Start: Minimal HuggingFace Inference Example (300m)

This is the simplest possible example showing the core inference flow.

Usage:
    python scripts/quickstart_hf_300m.py audio.wav
"""

import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2Config, LlamaConfig, Wav2Vec2Model, LlamaForCausalLM

# Import HF implementation
from omnilingual_asr.models.wav2vec2_llama.model_hf import (
    Wav2Vec2LlamaModelHF,
    VocabularyInfo,
)
from omnilingual_asr.models.wav2vec2_llama.config import ModelType
from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF


class MinimalTokenizer:
    """Minimal tokenizer for demo purposes."""
    def decode(self, tokens, skip_special_tokens=True):
        return f"[Transcription with {len(tokens)} tokens]"


def create_300m_model():
    """Create 300m model architecture with random weights."""

    # Model dimensions
    ENCODER_DIM = 1024
    DECODER_DIM = 4096
    VOCAB_SIZE = 9812

    # Create components
    encoder_config = Wav2Vec2Config(hidden_size=ENCODER_DIM, num_hidden_layers=24)
    decoder_config = LlamaConfig(hidden_size=DECODER_DIM, num_hidden_layers=12, vocab_size=VOCAB_SIZE)

    encoder = Wav2Vec2Model(encoder_config)
    decoder = LlamaForCausalLM(decoder_config)
    encoder_proj = torch.nn.Linear(ENCODER_DIM, DECODER_DIM)
    text_frontend = torch.nn.Embedding(VOCAB_SIZE + 1, DECODER_DIM)
    final_proj = torch.nn.Linear(DECODER_DIM, VOCAB_SIZE, bias=False)

    vocab_info = VocabularyInfo(size=VOCAB_SIZE, bos_idx=0, eos_idx=2, pad_idx=1)

    # Assemble model
    model = Wav2Vec2LlamaModelHF(
        model_type=ModelType.LLM_ASR,
        model_dim=DECODER_DIM,
        encoder=encoder,
        encoder_proj=encoder_proj,
        text_frontend=text_frontend,
        llama_decoder=decoder,
        final_proj=final_proj,
        target_vocab_info=vocab_info,
    )

    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: python quickstart_hf_300m.py audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    print("Creating 300m model...")
    model = create_300m_model()
    model.eval()

    print("Creating tokenizer...")
    tokenizer = MinimalTokenizer()

    print("Creating pipeline...")
    pipeline = ASRInferencePipelineHF(
        model=model,
        tokenizer=tokenizer,
        device="cpu",  # Use CPU for demo
        dtype=torch.float32,
    )

    print(f"Transcribing {audio_file}...")
    try:
        transcriptions = pipeline([audio_file])
        print(f"Result: {transcriptions[0]}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This is a demo with random weights.")
        print("For real inference, load a converted checkpoint.")


if __name__ == "__main__":
    main()
