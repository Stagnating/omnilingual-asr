# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration classes for HuggingFace-based models (no fairseq2 dependencies).

This module contains configuration classes that can be used without fairseq2.
For fairseq2-based training/inference, use config.py instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

WAV2VEC2_LLAMA_FAMILY: Final = "wav2vec2_llama"


class ModelType(Enum):
    """Model type enumeration."""
    LLM_ASR = 1
    LLM_ASR_LID = 2
    ZERO_SHOT = 3


@dataclass(kw_only=True)
class Wav2Vec2LlamaBeamSearchConfig:
    """Contains the settings for the LLM-ASR beam search."""

    nbest: int = 5
    """The size of the beam."""

    length_norm: bool = False
    """Whether we apply length normalization when computing hypothesis score."""

    compression_window: int = 100
    """For early stopping during decoding when inputs are bad, we try to compress the last `compression_window` tokens, and if the compression ratio is larger than `compression_threshold`, we stop decoding."""

    compression_threshold: float = 4.0
    """See `compression_window`."""
