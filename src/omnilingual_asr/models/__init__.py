# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Lazy imports to avoid loading fairseq2-dependent modules
# when using HuggingFace-only inference
# Import what you need explicitly, e.g.:
#   from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF

__all__ = [
    "inference",
]
