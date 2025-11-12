# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

__version__ = "0.1.0"

# Try to import fairseq2-dependent components
# These are only needed for training and fairseq2-based inference
# HuggingFace-based inference does not require fairseq2
try:
    from fairseq2.composition.assets import register_package_assets
    from fairseq2.composition.models import register_model_family
    from fairseq2.runtime.dependency import DependencyContainer

    from omnilingual_asr.models.wav2vec2_asr.config import (
        register_omnilingual_asr_wav2vec2_asr_configs,
    )
    from omnilingual_asr.models.wav2vec2_llama import (
        WAV2VEC2_LLAMA_FAMILY,
        Wav2Vec2LlamaConfig,
        Wav2Vec2LlamaModel,
        convert_wav2vec2_llama_state_dict,
        create_wav2vec2_llama_model,
        register_wav2vec2_llama_configs,
    )
    from omnilingual_asr.models.wav2vec2_ssl.config import (
        register_omnilingual_asr_wav2vec2_ssl_configs,
    )

    _FAIRSEQ2_AVAILABLE = True
except ImportError:
    _FAIRSEQ2_AVAILABLE = False
    # Fairseq2 not available - HuggingFace-only mode
    # The following will not be available:
    # - setup_fairseq2_extension()
    # - Fairseq2-based training/inference
    # But HuggingFace-based inference will work fine


def setup_fairseq2_extension(container: "DependencyContainer") -> None:
    """Setup fairseq2 extension. Only available when fairseq2 is installed."""
    if not _FAIRSEQ2_AVAILABLE:
        raise ImportError(
            "fairseq2 is required for setup_fairseq2_extension(). "
            "Install with: pip install fairseq2[arrow] "
            "Or use HuggingFace-based inference which does not require fairseq2."
        )

    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <omnilingual_asr>/cards.
    register_package_assets(container, "omnilingual_asr.cards")

    _register_models(container)


def _register_models(container: "DependencyContainer") -> None:
    """Register models with fairseq2. Only available when fairseq2 is installed."""
    if not _FAIRSEQ2_AVAILABLE:
        raise ImportError("fairseq2 is required for model registration")

    # Only adding custom wav2vec2 archs for wav2vec2_ssl model in fs2
    register_omnilingual_asr_wav2vec2_ssl_configs(container)

    # Only adding custom wav2vec2 archs for wav2vec2_asr model in fs2
    register_omnilingual_asr_wav2vec2_asr_configs(container)

    # wav2vec2 llama
    register_model_family(
        container,
        WAV2VEC2_LLAMA_FAMILY,
        kls=Wav2Vec2LlamaModel,
        config_kls=Wav2Vec2LlamaConfig,
        factory=create_wav2vec2_llama_model,
        state_dict_converter=convert_wav2vec2_llama_state_dict,
    )

    register_wav2vec2_llama_configs(container)
