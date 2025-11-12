# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tokenizer utilities for HuggingFace inference without fairseq2.

Provides wrappers for SentencePiece and other tokenizer formats
that work with the HuggingFace-based inference pipeline.
"""

from pathlib import Path
from typing import List, Union

import torch


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper compatible with HF inference pipeline.

    This provides the same interface as fairseq2 tokenizers but uses
    sentencepiece directly without fairseq2 dependencies.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        bos_idx: int = 0,
        eos_idx: int = 2,
        pad_idx: int = 1,
        unk_idx: int = 3,
    ):
        """
        Initialize SentencePiece tokenizer.

        Args:
            model_path: Path to .model file (SentencePiece model)
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
            pad_idx: Padding token index
            unk_idx: Unknown token index
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece is required for tokenization. "
                "Install with: pip install sentencepiece"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        # Special token indices
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        # Vocabulary info
        self.vocab_size = self.sp.vocab_size()

        print(f"Loaded SentencePiece tokenizer from {model_path}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Special tokens: BOS={bos_idx}, EOS={eos_idx}, PAD={pad_idx}, UNK={unk_idx}")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            List of token IDs
        """
        # Encode text
        token_ids = self.sp.encode(text, out_type=int)

        # Add special tokens
        if add_bos:
            token_ids = [self.bos_idx] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_idx]

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        # Filter special tokens if requested
        if skip_special_tokens:
            special_tokens = {self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx}
            token_ids = [t for t in token_ids if t not in special_tokens]

        # Decode
        text = self.sp.decode(token_ids)
        return text

    def __call__(
        self,
        text: Union[str, List[str]],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text (convenience method).

        Args:
            text: Input text or list of texts
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            Token IDs or list of token IDs
        """
        if isinstance(text, str):
            return self.encode(text, add_bos=add_bos, add_eos=add_eos)
        else:
            return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in text]


class HuggingFaceTokenizer:
    """Wrapper for HuggingFace tokenizers to work with inference pipeline.

    This allows using any HuggingFace tokenizer (e.g., from transformers library)
    with the inference pipeline.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        bos_idx: int = 0,
        eos_idx: int = 2,
        pad_idx: int = 1,
        unk_idx: int = 3,
    ):
        """
        Initialize HuggingFace tokenizer.

        Args:
            tokenizer_name_or_path: HF tokenizer name or path
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
            pad_idx: Padding token index
            unk_idx: Unknown token index
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # Special token indices
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        # Vocabulary size
        self.vocab_size = len(self.tokenizer)

        print(f"Loaded HuggingFace tokenizer: {tokenizer_name_or_path}")
        print(f"  Vocabulary size: {self.vocab_size}")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode text to token IDs."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if add_bos:
            token_ids = [self.bos_idx] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_idx]

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        if skip_special_tokens:
            special_tokens = {self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx}
            token_ids = [t for t in token_ids if t not in special_tokens]

        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text

    def __call__(
        self,
        text: Union[str, List[str]],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Union[List[int], List[List[int]]]:
        """Encode text (convenience method)."""
        if isinstance(text, str):
            return self.encode(text, add_bos=add_bos, add_eos=add_eos)
        else:
            return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in text]


def load_tokenizer(
    tokenizer_path: Union[str, Path],
    tokenizer_type: str = "sentencepiece",
    **kwargs,
):
    """
    Load a tokenizer from file.

    Args:
        tokenizer_path: Path to tokenizer file (.model for sentencepiece)
                       or HF tokenizer name/path
        tokenizer_type: Type of tokenizer ("sentencepiece" or "huggingface")
        **kwargs: Additional arguments passed to tokenizer constructor

    Returns:
        Tokenizer instance

    Examples:
        # Load SentencePiece tokenizer
        tokenizer = load_tokenizer("tokenizer.model", "sentencepiece")

        # Load HuggingFace tokenizer
        tokenizer = load_tokenizer("bert-base-uncased", "huggingface")
    """
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(tokenizer_path, **kwargs)
    elif tokenizer_type == "huggingface":
        return HuggingFaceTokenizer(tokenizer_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Supported types: 'sentencepiece', 'huggingface'"
        )
