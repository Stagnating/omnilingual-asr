# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from omnilingual_asr.models.wav2vec2_llama.beamsearch_hf import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF,
)
from omnilingual_asr.models.wav2vec2_llama.config import (
    ModelType,
    Wav2Vec2LlamaBeamSearchConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model_hf import (
    Wav2Vec2LlamaModelHF,
    VocabularyInfo,
)
from omnilingual_asr.models.inference.audio_utils import (
    load_audio,
    resample_audio,
    validate_audio_length,
    collate_audio_batch,
)

AudioInput = (
    List[Path]
    | List[str]
    | List[str | Path]
    | List[bytes]
    | List[NDArray[np.int8]]
    | List[bytes | NDArray[np.int8]]
    | List[Dict[str, Any]]
)

MAX_ALLOWED_AUDIO_SEC: Final = 40


class ASRInferencePipelineHF:
    """HuggingFace-based ASR inference pipeline without fairseq2 dependencies.

    This is a drop-in replacement for ASRInferencePipeline that uses
    HuggingFace transformers instead of fairseq2.
    """

    def __init__(
        self,
        model: Wav2Vec2LlamaModelHF,
        tokenizer: Any,  # Should be a tokenizer with encode/decode methods
        device: str | None | torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        beam_search_config: Wav2Vec2LlamaBeamSearchConfig | None = None,
    ) -> None:
        """
        Initialize the HuggingFace-based inference pipeline.

        Args:
            model: Pre-loaded Wav2Vec2LlamaModelHF instance
            tokenizer: Tokenizer instance with encode/decode methods
            device: Device to run inference on
            dtype: Data type for model inference
            beam_search_config: Optional beam search configuration
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        # Set model
        self.model = model
        self.model = self.model.to(device=self.device)
        self.model.eval()

        # Set tokenizer
        self.tokenizer = tokenizer

        # Set up beam search
        if beam_search_config is None:
            beam_search_config = Wav2Vec2LlamaBeamSearchConfig(
                nbest=1,
                length_norm=False,
            )

        self.beam_search_generator = None
        if isinstance(self.model, Wav2Vec2LlamaModelHF):
            self.beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF(
                model=self.model, config=beam_search_config
            )

        print(f"Pipeline initialized on {self.device} with dtype {self.dtype}")

    def _create_batch(
        self, wavs_langs: List[Tuple[torch.Tensor, str | None]]
    ) -> Dict:
        """Create a batch dict from audio tensors."""
        # Create audio data structure
        audio_examples = []
        for item in wavs_langs:
            audio_examples.append(
                {
                    "audio_feature": item[0],
                    "text": torch.tensor(
                        [0], dtype=torch.int64
                    ),  # Dummy text for inference
                }
            )

        # Collate the examples
        collated_data = collate_audio_batch(audio_examples)

        # Extract audio and text data
        audio_data = collated_data["audio_feature"]
        text_data = collated_data["text"]

        example = {"lang": [item[1] for item in wavs_langs]}
        if all(x is None for x in example["lang"]):
            example = {}

        # Create batch dict (HF format)
        return {
            'source_seqs': audio_data["seqs"].to(self.device, self.dtype),
            'source_seq_lens': audio_data["seq_lens"],
            'target_seqs': text_data["seqs"].to(self.device),
            'target_seq_lens': text_data["seq_lens"],
            'example': example,
        }

    def _apply_model(self, batch: Dict) -> List[str]:
        """Apply model forward pass to the batch."""
        # Get context embeddings from model
        context_logits, context_lengths = self.model(batch, return_decoder_inputs=True)

        # Generate hypotheses using beam search
        assert self.beam_search_generator is not None
        hypothesis_tokens, hypothesis_lengths = (
            self.beam_search_generator.generate_hypotheses(
                decoder_context_inputs=context_logits,
                decoder_context_lengths=context_lengths,
            )
        )

        # Decode tokens to text
        transcriptions = []
        for i in range(hypothesis_tokens.shape[0]):
            seq_len = hypothesis_lengths[i]
            tokens = hypothesis_tokens[i, :seq_len]
            # Use tokenizer's decode method
            text = self.tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)
            transcriptions.append(text)

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return transcriptions

    def process_audio_input(
        self, audio_input: str | Path | bytes | NDArray[np.int8] | Dict[str, Any]
    ) -> torch.Tensor:
        """Process a single audio input into a tensor.

        Args:
            audio_input: Audio in various formats

        Returns:
            Audio tensor ready for model input
        """
        # Load and resample audio
        audio_data = load_audio(audio_input, target_sample_rate=16000)

        # Validate length
        audio_data = validate_audio_length(
            audio_data, max_seconds=MAX_ALLOWED_AUDIO_SEC
        )

        # Extract waveform and convert to tensor
        waveform = audio_data["waveform"]

        # Ensure it's a 1D tensor
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if waveform.dim() == 2:
            # Take first channel if stereo
            waveform = waveform[0]

        return waveform

    def __call__(
        self,
        audio: AudioInput,
        lang: str | List[str] | None = None,
    ) -> List[str]:
        """
        Run ASR inference on audio inputs.

        Args:
            audio: List of audio inputs in various formats
            lang: Optional language code(s) for language-aware models

        Returns:
            List of transcriptions
        """
        # Normalize inputs to lists
        if not isinstance(audio, list):
            audio = [audio]

        # Process language codes
        if lang is None:
            langs = [None] * len(audio)
        elif isinstance(lang, str):
            langs = [lang] * len(audio)
        else:
            langs = lang
            assert len(langs) == len(audio), "Number of languages must match number of audio inputs"

        # Process all audio inputs
        audio_tensors = []
        for audio_input in audio:
            try:
                tensor = self.process_audio_input(audio_input)
                audio_tensors.append(tensor)
            except Exception as e:
                raise RuntimeError(f"Failed to process audio input: {e}") from e

        # Create batch
        wavs_langs = list(zip(audio_tensors, langs))
        batch = self._create_batch(wavs_langs)

        # Run model
        with torch.no_grad():
            transcriptions = self._apply_model(batch)

        return transcriptions

    def transcribe(
        self,
        audio: AudioInput,
        lang: str | List[str] | None = None,
    ) -> List[str]:
        """Alias for __call__ for compatibility."""
        return self(audio, lang)
