# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Lazy imports to avoid loading fairseq2 dependencies
# Import what you need explicitly:
#   For fairseq2-based inference:
#     from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
#   For HuggingFace-based inference:
#     from omnilingual_asr.models.inference.pipeline_hf import ASRInferencePipelineHF

__all__ = [
    "ContextExample",
    "ASRInferencePipeline",
    "ASRInferencePipelineHF",
]
