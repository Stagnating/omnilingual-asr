#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert fairseq2 checkpoints to HuggingFace format.

This script loads a fairseq2 model checkpoint and converts it to a format
compatible with the HuggingFace-based Wav2Vec2LlamaModelHF.

Usage:
    python scripts/convert_fairseq2_to_hf.py \
        --input_checkpoint path/to/fairseq2/checkpoint.pt \
        --output_dir path/to/output/dir \
        --model_config 7b

The output directory will contain:
    - pytorch_model.bin: Converted model weights
    - config.json: Model configuration (if applicable)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import Wav2Vec2Config, LlamaConfig


def convert_encoder_weights(fs2_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Wav2Vec2 encoder weights from fairseq2 to HuggingFace format.

    Args:
        fs2_state_dict: fairseq2 state dictionary

    Returns:
        HuggingFace-compatible state dictionary for Wav2Vec2
    """
    hf_state_dict = {}

    # Mapping rules for Wav2Vec2
    # fairseq2: encoder_frontend.* -> HF: wav2vec2.feature_extractor.*
    # fairseq2: encoder.* -> HF: wav2vec2.encoder.*

    for key, value in fs2_state_dict.items():
        new_key = key

        # Convert encoder frontend (feature extractor)
        if key.startswith("encoder_frontend."):
            # fairseq2's Wav2Vec2Frontend -> HF's feature_extractor
            new_key = key.replace("encoder_frontend.", "wav2vec2.feature_extractor.")

            # Specific layer mappings
            new_key = new_key.replace("post_extract_layer_norm", "layer_norm")
            new_key = new_key.replace("model_dim_proj", "projection")

        # Convert encoder layers
        elif key.startswith("encoder."):
            # fairseq2's TransformerEncoder -> HF's encoder
            new_key = key.replace("encoder.", "wav2vec2.encoder.")

            # Layer-specific mappings
            new_key = new_key.replace("layers.", "layers.")
            new_key = new_key.replace("self_attn.", "attention.")
            new_key = new_key.replace("self_attn_layer_norm", "layer_norm")
            new_key = new_key.replace("ffn.", "feed_forward.")
            new_key = new_key.replace("ffn_layer_norm", "final_layer_norm")

        hf_state_dict[new_key] = value

    return hf_state_dict


def convert_decoder_weights(fs2_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert LLaMA decoder weights from fairseq2 to HuggingFace format.

    Args:
        fs2_state_dict: fairseq2 state dictionary

    Returns:
        HuggingFace-compatible state dictionary for LLaMA
    """
    hf_state_dict = {}

    # Mapping rules for LLaMA decoder
    # fairseq2: llama_decoder.* -> HF: model.* (LlamaForCausalLM structure)

    for key, value in fs2_state_dict.items():
        new_key = key

        if key.startswith("llama_decoder."):
            # Remove llama_decoder prefix and map to HF structure
            new_key = key.replace("llama_decoder.", "model.")

            # Layer-specific mappings
            new_key = new_key.replace("layers.", "layers.")
            new_key = new_key.replace("self_attn.", "self_attn.")
            new_key = new_key.replace("self_attn_layer_norm", "input_layernorm")
            new_key = new_key.replace("ffn.", "mlp.")
            new_key = new_key.replace("ffn_layer_norm", "post_attention_layernorm")

            # Attention mappings
            new_key = new_key.replace("q_proj", "q_proj")
            new_key = new_key.replace("k_proj", "k_proj")
            new_key = new_key.replace("v_proj", "v_proj")
            new_key = new_key.replace("output_proj", "o_proj")

            # FFN mappings
            new_key = new_key.replace("gate_proj", "gate_proj")
            new_key = new_key.replace("fc1", "up_proj")
            new_key = new_key.replace("fc2", "down_proj")

        hf_state_dict[new_key] = value

    return hf_state_dict


def convert_checkpoint(
    input_path: Path,
    output_dir: Path,
    model_config: str = "7b",
) -> None:
    """Convert a complete fairseq2 checkpoint to HuggingFace format.

    Args:
        input_path: Path to fairseq2 checkpoint (.pt file)
        output_dir: Directory to save converted checkpoint
        model_config: Model configuration name (e.g., "7b", "300m")
    """
    print(f"Loading fairseq2 checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Extract model state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Checkpoint contains {len(state_dict)} parameters")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert weights
    converted_state_dict = {}

    # 1. Convert encoder (Wav2Vec2)
    print("Converting encoder weights...")
    encoder_weights = {k: v for k, v in state_dict.items()
                      if k.startswith("encoder_frontend.") or k.startswith("encoder.")}
    converted_encoder = convert_encoder_weights(encoder_weights)
    converted_state_dict.update(converted_encoder)

    # 2. Convert decoder (LLaMA)
    print("Converting decoder weights...")
    decoder_weights = {k: v for k, v in state_dict.items()
                      if k.startswith("llama_decoder.")}
    converted_decoder = convert_decoder_weights(decoder_weights)
    converted_state_dict.update(converted_decoder)

    # 3. Copy other weights directly (encoder_proj, text_frontend, final_proj, etc.)
    print("Copying other weights...")
    other_keys = [
        "encoder_proj.weight", "encoder_proj.bias",
        "text_frontend.weight",
        "final_proj.weight",
        "lang_embeddings.weight",
    ]

    for key in state_dict:
        if any(key.startswith(ok.split('.')[0]) for ok in other_keys):
            if key not in converted_state_dict:
                converted_state_dict[key] = state_dict[key]

    print(f"Converted {len(converted_state_dict)} parameters")

    # Save converted checkpoint
    output_path = output_dir / "pytorch_model.bin"
    print(f"Saving converted checkpoint to {output_path}")
    torch.save(converted_state_dict, output_path)

    # Save metadata
    metadata = {
        "source_checkpoint": str(input_path),
        "model_config": model_config,
        "num_parameters": len(converted_state_dict),
        "conversion_info": {
            "encoder_params": len(encoder_weights),
            "decoder_params": len(decoder_weights),
        }
    }

    metadata_path = output_dir / "conversion_metadata.json"
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Conversion complete!")
    print(f"\nOutput files:")
    print(f"  - Model weights: {output_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"\nTo load the converted model:")
    print(f"  state_dict = torch.load('{output_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert fairseq2 checkpoints to HuggingFace format"
    )
    parser.add_argument(
        "--input_checkpoint",
        type=Path,
        required=True,
        help="Path to fairseq2 checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save converted checkpoint",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="7b",
        help="Model configuration name (e.g., 7b, 300m, 1b, 3b)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_checkpoint.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input_checkpoint}")

    # Run conversion
    convert_checkpoint(
        input_path=args.input_checkpoint,
        output_dir=args.output_dir,
        model_config=args.model_config,
    )


if __name__ == "__main__":
    main()
