# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Audio processing utilities to replace fairseq2 AudioDecoder and DataPipeline."""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from numpy.typing import NDArray


def load_audio(
    audio_input: Union[str, Path, bytes, NDArray[np.int8], Dict[str, Any]],
    target_sample_rate: int = 16000,
) -> Dict[str, Any]:
    """
    Load audio from various input formats and resample to target sample rate.

    Args:
        audio_input: Can be:
            - Path to audio file (str or Path)
            - Raw bytes
            - NumPy array
            - Dict with 'waveform' and 'sample_rate' keys
        target_sample_rate: Target sample rate (default: 16000)

    Returns:
        Dictionary with 'waveform' (Tensor) and 'sample_rate' (int) keys
    """
    # If already a dict with waveform, just validate/resample
    if isinstance(audio_input, dict):
        if 'waveform' in audio_input and 'sample_rate' in audio_input:
            return resample_audio(audio_input, target_sample_rate)
        else:
            raise ValueError("Dict input must contain 'waveform' and 'sample_rate' keys")

    # Handle file paths
    if isinstance(audio_input, (str, Path)):
        try:
            waveform, sample_rate = torchaudio.load(str(audio_input))
        except RuntimeError as e:
            # Fallback to soundfile if torchaudio backend fails (common on Windows)
            if "Couldn't find appropriate backend" in str(e):
                try:
                    import soundfile as sf
                    data, sample_rate = sf.read(str(audio_input), dtype='float32')
                    # Convert to tensor and ensure shape is [channels, time]
                    waveform = torch.from_numpy(data).float()
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)  # [time] -> [1, time]
                    else:
                        # soundfile returns [time, channels], transpose to [channels, time]
                        waveform = waveform.transpose(0, 1)
                except ImportError:
                    raise RuntimeError(
                        f"Failed to load audio file {audio_input}. "
                        "torchaudio backend is not available on your system. "
                        "Please install soundfile: pip install soundfile"
                    ) from e
            else:
                raise

        audio_data = {
            'waveform': waveform,
            'sample_rate': sample_rate,
        }
        return resample_audio(audio_data, target_sample_rate)

    # Handle raw bytes
    if isinstance(audio_input, bytes):
        # Save to temporary file and load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_input)
            tmp_path = tmp.name

        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
            audio_data = {
                'waveform': waveform,
                'sample_rate': sample_rate,
            }
            return resample_audio(audio_data, target_sample_rate)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Handle numpy array
    if isinstance(audio_input, np.ndarray):
        # Assume numpy array is already at target sample rate
        # Convert to tensor
        waveform = torch.from_numpy(audio_input).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        audio_data = {
            'waveform': waveform,
            'sample_rate': target_sample_rate,
        }
        return audio_data

    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")


def resample_audio(
    audio_data: Dict[str, Any], target_sample_rate: int = 16000
) -> Dict[str, Any]:
    """
    Resample audio waveform to target sample rate.

    Args:
        audio_data: Dictionary containing 'waveform' and 'sample_rate' keys
        target_sample_rate: Target sample rate (default: 16000)

    Returns:
        Dictionary with resampled waveform and updated sample rate
    """
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data["sample_rate"]

    if current_sample_rate != target_sample_rate:
        # Ensure waveform is in the correct shape [channels, time]
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        # Handle different shapes
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [time] -> [1, time]
        elif waveform.dim() == 2:
            # Heuristic: if first dim > second dim, transpose
            if waveform.shape[0] > waveform.shape[1]:
                waveform = waveform.transpose(0, 1)

        # Resample
        waveform = F.resample(
            waveform,
            orig_freq=current_sample_rate,
            new_freq=target_sample_rate,
        )

        audio_data["sample_rate"] = target_sample_rate
        audio_data["waveform"] = waveform

    return audio_data


def collate_audio_batch(
    batch: List[Dict[str, Any]], pad_value: float = 0.0
) -> Dict[str, Any]:
    """
    Collate a batch of audio examples with padding.

    Args:
        batch: List of dictionaries, each containing:
            - 'audio_feature': Tensor of shape [T] or [C, T]
            - 'text': Tensor of token IDs (optional)
            - Other fields...
        pad_value: Value to use for padding

    Returns:
        Dictionary with collated and padded tensors:
            - 'audio_feature': Dict with 'seqs' [B, T] and 'seq_lens' List[int]
            - 'text': Dict with 'seqs' [B, S] and 'seq_lens' List[int]
            - Other fields from the first example
    """
    # Extract audio features
    audio_features = [item['audio_feature'] for item in batch]

    # Ensure all are 1D (squeeze if needed)
    audio_features = [
        f.squeeze() if f.dim() > 1 else f for f in audio_features
    ]

    # Get lengths
    audio_lens = [f.shape[0] for f in audio_features]

    # Pad audio to max length
    max_audio_len = max(audio_lens)
    padded_audio = torch.stack([
        torch.nn.functional.pad(
            f, (0, max_audio_len - f.shape[0]), value=pad_value
        )
        for f in audio_features
    ])

    result = {
        'audio_feature': {
            'seqs': padded_audio,
            'seq_lens': audio_lens,
        }
    }

    # Collate text if present
    if 'text' in batch[0]:
        text_tokens = [item['text'] for item in batch]
        text_lens = [t.shape[0] for t in text_tokens]
        max_text_len = max(text_lens)

        # Pad text (usually with pad_idx=1, but we'll use 0 for now)
        padded_text = torch.stack([
            torch.nn.functional.pad(
                t, (0, max_text_len - t.shape[0]), value=0
            )
            for t in text_tokens
        ])

        result['text'] = {
            'seqs': padded_text,
            'seq_lens': text_lens,
        }

    # Copy other fields from first example
    for key in batch[0]:
        if key not in result and key not in ['audio_feature', 'text']:
            result[key] = batch[0][key]

    return result


def validate_audio_length(
    audio_data: Dict[str, Any],
    max_seconds: float = 40.0,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """
    Validate that audio length doesn't exceed maximum duration.

    Args:
        audio_data: Dictionary containing 'waveform' and 'sample_rate'
        max_seconds: Maximum allowed duration in seconds
        sample_rate: Sample rate to use for length calculation

    Returns:
        Same audio_data if valid

    Raises:
        ValueError: If audio exceeds maximum length
    """
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data.get("sample_rate", sample_rate)

    # Calculate length in seconds
    if isinstance(waveform, torch.Tensor):
        num_samples = waveform.shape[-1]  # Last dimension is time
    elif isinstance(waveform, np.ndarray):
        num_samples = waveform.shape[-1]
    else:
        raise ValueError(f"Unsupported waveform type: {type(waveform)}")

    duration_seconds = num_samples / current_sample_rate

    if duration_seconds > max_seconds:
        raise ValueError(
            f"Audio duration ({duration_seconds:.2f}s) exceeds maximum "
            f"allowed length ({max_seconds}s)"
        )

    return audio_data
