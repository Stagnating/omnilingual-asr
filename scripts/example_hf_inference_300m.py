#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example: HuggingFace Inference with 300m Model

This script demonstrates how to use the HuggingFace-based inference pipeline
without fairseq2 dependencies for the 300m model size.

Usage:
    python scripts/example_hf_inference_300m.py \
        --checkpoint_dir path/to/converted/checkpoint \
        --tokenizer_path path/to/tokenizer.model \
        --audio_files audio1.wav audio2.wav \
        --lang en

Before running:
    1. Download the tokenizer:
       python scripts/download_tokenizer.py \
           --output tokenizer.model \
           --model-config 300m

    2. Convert your fairseq2 checkpoint:
       python scripts/convert_fairseq2_to_hf.py \
           --input_checkpoint your_300m_checkpoint.pt \
           --output_dir converted_300m \
           --model_config 300m \
           --tokenizer_path tokenizer.model

    3. Install dependencies:
       pip install transformers accelerate sentencepiece torch torchaudio
"""

import argparse
from pathlib import Path
from typing import List

import torch
from transformers import Wav2Vec2Config, LlamaConfig, Wav2Vec2Model, LlamaForCausalLM

from omnilingual_asr.models.wav2vec2_llama.model_hf import (
    Wav2Vec2LlamaModelHF,
    VocabularyInfo,
)
from omnilingual_asr.models.wav2vec2_llama.config_hf import (
    ModelType,
    Wav2Vec2LlamaBeamSearchConfig,
)
from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF
from omnilingual_asr.models.inference.tokenizer import load_tokenizer


def load_300m_model(
    checkpoint_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Wav2Vec2LlamaModelHF:
    """
    Load the 300m HuggingFace model from a converted checkpoint.

    Args:
        checkpoint_path: Path to pytorch_model.bin (converted checkpoint)
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Loaded Wav2Vec2LlamaModelHF model
    """
    print(f"Loading 300m model from {checkpoint_path}")

    # 300m Model Configuration
    # These are the standard dimensions for the 300m model
    ENCODER_DIM = 1024      # Wav2Vec2 hidden size
    DECODER_DIM = 4096      # LLaMA hidden size
    VOCAB_SIZE = 9812       # Vocabulary size

    # Step 1: Configure Wav2Vec2 Encoder
    encoder_config = Wav2Vec2Config(
        hidden_size=ENCODER_DIM,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
    )

    # Step 2: Configure LLaMA Decoder (300m config)
    decoder_config = LlamaConfig(
        hidden_size=DECODER_DIM,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.1,
        pad_token_id=1,
    )

    print("Creating model architecture...")

    # Step 3: Initialize base models
    encoder = Wav2Vec2Model(encoder_config)
    decoder = LlamaForCausalLM(decoder_config)

    # Step 4: Create projection and embedding layers
    encoder_proj = torch.nn.Linear(ENCODER_DIM, DECODER_DIM, bias=True)
    text_frontend = torch.nn.Embedding(VOCAB_SIZE + 1, DECODER_DIM)  # +1 for special token
    final_proj = torch.nn.Linear(DECODER_DIM, VOCAB_SIZE, bias=False)

    # Step 5: Create vocabulary info
    vocab_info = VocabularyInfo(
        size=VOCAB_SIZE,
        unk_idx=3,
        bos_idx=0,
        eos_idx=2,
        pad_idx=1,
    )

    # Step 6: Assemble the complete model
    model = Wav2Vec2LlamaModelHF(
        model_type=ModelType.LLM_ASR_LID,  # 300m uses LID model type
        model_dim=DECODER_DIM,
        encoder=encoder,
        encoder_proj=encoder_proj,
        text_frontend=text_frontend,
        llama_decoder=decoder,
        final_proj=final_proj,
        target_vocab_info=vocab_info,
        max_generation_length=8192,
        encoder_stacking=1,
        lang_embeddings_p=0.5,  # Language embedding probability
        language_column_name="lang",
    )

    # Step 7: Load converted weights
    if checkpoint_path.exists():
        print("Loading checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load with strict=False to handle missing/extra keys
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) < 10:
                for key in missing_keys:
                    print(f"  - {key}")

        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) < 10:
                for key in unexpected_keys:
                    print(f"  - {key}")

        print("✓ Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Initializing with random weights (for testing only)")

    # Step 8: Move to device and set dtype
    print(f"Moving model to {device} with dtype {dtype}")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params / 1e6:.1f}M parameters")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Example HuggingFace inference with 300m model"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing converted checkpoint (pytorch_model.bin)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer model file (.model for SentencePiece)",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="sentencepiece",
        choices=["sentencepiece", "huggingface"],
        help="Type of tokenizer (default: sentencepiece)",
    )
    parser.add_argument(
        "--audio_files",
        type=str,
        nargs="+",
        required=True,
        help="Audio files to transcribe",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es', 'fr')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model inference",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search (1 = greedy)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Validate checkpoint directory
    checkpoint_path = args.checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nDid you convert your checkpoint? Run:")
        print("  python scripts/convert_fairseq2_to_hf.py \\")
        print("      --input_checkpoint your_checkpoint.pt \\")
        print(f"      --output_dir {args.checkpoint_dir} \\")
        print("      --model_config 300m")
        return

    # Load model
    print("\n" + "="*60)
    print("STEP 1: Loading Model")
    print("="*60)
    model = load_300m_model(checkpoint_path, device=args.device, dtype=dtype)

    # Load tokenizer
    print("\n" + "="*60)
    print("STEP 2: Loading Tokenizer")
    print("="*60)

    try:
        tokenizer = load_tokenizer(
            tokenizer_path=args.tokenizer_path,
            tokenizer_type=args.tokenizer_type,
            bos_idx=0,
            eos_idx=2,
            pad_idx=1,
            unk_idx=3,
        )
        print("✓ Tokenizer loaded successfully")
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {args.tokenizer_path}")
        print("\nMake sure you have the tokenizer file (.model for SentencePiece)")
        print("It should be available alongside your model checkpoint.")
        return
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the required tokenizer library:")
        if args.tokenizer_type == "sentencepiece":
            print("  pip install sentencepiece")
        else:
            print("  pip install transformers")
        return

    # Configure beam search
    beam_config = Wav2Vec2LlamaBeamSearchConfig(
        nbest=args.beam_size,
        length_norm=False,
        compression_window=100,
        compression_threshold=4.0,
    )
    print(f"✓ Beam search configured (beam_size={args.beam_size})")

    # Create inference pipeline
    print("\n" + "="*60)
    print("STEP 3: Creating Inference Pipeline")
    print("="*60)
    pipeline = ASRInferencePipelineHF(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype,
        beam_search_config=beam_config,
    )
    print("✓ Pipeline ready")

    # Run inference
    print("\n" + "="*60)
    print("STEP 4: Running Inference")
    print("="*60)

    print(f"Transcribing {len(args.audio_files)} audio file(s)...")
    if args.lang:
        print(f"Language hint: {args.lang}")

    try:
        transcriptions = pipeline(
            audio=args.audio_files,
            lang=args.lang,
        )

        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        for audio_file, transcription in zip(args.audio_files, transcriptions):
            print(f"\n📄 {audio_file}")
            print(f"📝 {transcription}")

        print("\n✓ Inference completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return

    # Memory info
    if args.device == "cuda":
        print(f"\n🔧 GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
