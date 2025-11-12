# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import zlib
from typing import Tuple, final, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig
from omnilingual_asr.models.wav2vec2_llama.model_hf import Wav2Vec2LlamaModelHF


@final
class Wav2Vec2LlamaBeamSearchSeq2SeqGeneratorHF:
    """HuggingFace-based beam search generator for ``Wav2Vec2LLamaModel`` speech-to-text models.

    Performs beam search decoding by maintaining multiple hypothesis candidates,
    prefilling with audio context embeddings, then iteratively generating tokens
    while tracking scores and handling early stopping via compression ratio analysis.

    This version uses HuggingFace's past_key_values instead of fairseq2's IncrementalStateBag.
    """

    def __init__(
        self,
        model: Wav2Vec2LlamaModelHF,
        config: Wav2Vec2LlamaBeamSearchConfig,
    ) -> None:
        self.model = model
        self.pad_idx = model.target_vocab_info.pad_idx
        self.eos_idx = model.target_vocab_info.eos_idx
        self.bos_idx = model.target_vocab_info.bos_idx
        self.config = config

    @staticmethod
    def idx_1d_to_2d(idx: Tensor, dim2: int) -> tuple[Tensor, Tensor]:
        return idx // dim2, idx % dim2

    @staticmethod
    def compression_ratio(text: str) -> float:
        text_bytes = text.encode("utf-8")
        return len(text_bytes) / len(zlib.compress(text_bytes))

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder the past_key_values cache based on beam indices.

        This is necessary when beam search selects different parent hypotheses.
        """
        if past_key_values is None:
            return None

        reordered_past = ()
        for layer_past in past_key_values:
            # Each layer_past is a tuple of (key, value) tensors
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @torch.no_grad()
    def generate_hypotheses(
        self,
        decoder_context_inputs: Tensor,
        decoder_context_lengths: List[int],
    ) -> Tuple[Tensor, List[int]]:
        """
        Conducts a beam search to generate hypotheses for the LLM-ASR model.

        Initialization and preparation:
        Prefill the decoder_input matrix with the embeddings of context already computed by the model,
        which is everything up to but excluding the <BOS> token. Feed the decoder the precomputed context.

        Generation Loop:
        Run the decoder on the latest generated token (or <BOS> at the first iterations), get emission scores.
        Add the log probability scores to the current scores of the hypotheses, choose the new `nbest` hypotheses.
        Don't change scores of hypos that already ended (emitted <EOS>). Decoding is done when all `nbest`
        hypotheses end with <EOS>.

        Args:
            decoder_context_inputs (Tensor):
                The input tensor containing the previously embedded context / prompt for the decoder.
                Everything up to and including the <BOS> token. Shape: [B, T, D]
            decoder_context_lengths (List[int]):
                The lengths of each sequence in the batch.

        Returns:
            Tuple[Tensor, List[int]]:
                A tuple containing:
                    - hypotheses (Tensor): The generated tokens for the batch. Shape: [B, S]
                    - lengths (List[int]): The length of the generated tokens in the batch.
        """
        # Some init
        B = decoder_context_inputs.size(0)
        device = decoder_context_inputs.device
        dtype = decoder_context_inputs.dtype
        nbest = self.config.nbest
        ex_separator = torch.arange(B, device=device).unsqueeze(1) * nbest

        # Prepare a decoder input matrix, prefill with context
        decoder_inputs = torch.zeros(
            [
                B * nbest,
                self.model.max_generation_length,
                self.model.model_dim,
            ],
            device=device,
            dtype=dtype,
        )
        decoder_inputs[:, : decoder_context_inputs.size(1)] = (
            decoder_context_inputs.repeat_interleave(nbest, dim=0)
        )
        context_lengths_tensor = torch.tensor(
            decoder_context_lengths, device=device, dtype=torch.long
        )
        context_lengths = context_lengths_tensor.repeat_interleave(nbest)

        # Prepare a token matrix and a scores matrix
        assert self.pad_idx is not None, "`pad_idx` must be specified"
        out_tokens = torch.full(
            [B * nbest, self.model.max_generation_length],
            fill_value=self.pad_idx,
            dtype=torch.int64,
            device=device,
        )
        scores = torch.zeros(B * nbest, dtype=torch.float, device=device) - 1e6
        scores[::nbest] = 0.0

        # Prefill with shortest context, keep cache (past_key_values)
        min_context_len = int(context_lengths.min()) - 1  # remove double BOS input
        prefill_seqs = decoder_inputs[:, :min_context_len]

        # Create attention mask for prefill
        attention_mask = torch.ones(
            B * nbest, min_context_len, device=device, dtype=torch.long
        )

        # Run prefill through decoder
        decoder_outputs = self.model.llama_decoder.model(
            inputs_embeds=prefill_seqs,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=None,
        )
        past_key_values = decoder_outputs.past_key_values

        # Iterative decoding:
        # Start decoding after the shortest context in the batch. Samples with longer
        # context will be ignored until their context has been consumed.
        # For each sample, choose either context, or emitted text embedding.
        # If EOS is emitted, the sample is non-active. Stop when there are no active samples.
        eos_mask = torch.zeros(B * nbest, dtype=torch.bool, device=device)
        done = False
        t = min_context_len

        # Build initial attention mask
        attention_mask = torch.ones(B * nbest, min_context_len + 1, device=device, dtype=torch.long)

        while not done:
            # Run the decoder on mixed context and emitted text embeddings
            iterative_seqs = decoder_inputs[:, t : t + 1]

            # Run decoder with cache
            decoder_outputs = self.model.llama_decoder.model(
                inputs_embeds=iterative_seqs,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            dec_out = decoder_outputs.last_hidden_state
            past_key_values = decoder_outputs.past_key_values

            # Project to vocabulary logits
            logits = self.model.final_proj(dec_out).squeeze(1)  # [B * nbest, V]
            log_probs = F.log_softmax(logits, dim=-1)

            # Choose nbest
            if self.config.length_norm:
                n_tokens = torch.logical_and(
                    out_tokens[:, :t] != self.pad_idx, out_tokens[:, :t] != self.eos_idx  # type: ignore[arg-type]
                ).sum(dim=1, keepdim=True)
                n_tokens = torch.clamp(n_tokens, min=1)  # Avoid division by zero
                candidate_scores = (scores.unsqueeze(1) * n_tokens + log_probs) / (
                    n_tokens + 1
                )
            else:
                candidate_scores = scores.unsqueeze(1) + log_probs  # [B * nbest, V]

            candidate_scores[eos_mask] = -torch.inf
            candidate_scores[eos_mask, self.eos_idx] = scores[
                eos_mask
            ]  # Don't change scores for ended hypos

            top_scores, top_idx = candidate_scores.view(B, -1).topk(
                k=nbest, dim=-1, sorted=True
            )
            top_idx_nbest, top_idx_v = self.idx_1d_to_2d(
                top_idx, candidate_scores.size(-1)
            )
            top_idx_b = (top_idx_nbest + ex_separator).view(-1)  # Parent hypos indices

            # Reorder some tensors based on parent hypos
            out_tokens = out_tokens[top_idx_b]
            eos_mask = eos_mask[top_idx_b]
            scores = scores[top_idx_b]

            # Reorder the cache based on selected beams
            past_key_values = self._reorder_cache(past_key_values, top_idx_b)

            scores = torch.where(eos_mask, scores, top_scores.view(-1))  # [B * nbest]
            out_tokens[:, t] = top_idx_v.view(-1)

            # For hypos that still don't emit tokens, set new tokens to pad_idx, score to 0.
            no_token_mask = t < context_lengths - 1
            out_tokens[no_token_mask, t] = self.pad_idx
            scores[no_token_mask] = 0.0

            # For hypos that had EOS previously, set new tokens to EOS. Scores don't change.
            # Set new EOS mask.
            assert self.eos_idx is not None, "`eos_idx` must be set"
            out_tokens[eos_mask, t] = self.eos_idx
            new_tokens = out_tokens[:, t : t + 1]
            eos_mask = (new_tokens == self.eos_idx).squeeze(1)

            # Run new tokens through frontend, set in decoder input
            new_tokens_embedded = self.model.embed_text(new_tokens, dtype=dtype)
            decoder_inputs[~no_token_mask, t + 1] = (
                new_tokens_embedded[~no_token_mask].to(decoder_inputs.dtype).squeeze(1)
            )  # Don't override audio encoder outputs

            # Update attention mask for next iteration
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(B * nbest, 1, device=device, dtype=torch.long)
            ], dim=1)

            # Early stopping if emitting repeating characters, use compression ratio
            # only every 250 steps, only when started emitting tokens more than compression_window tokens ago
            if t % 250 == 0 and t > min_context_len + self.config.compression_window:
                cpu_tokens = (
                    out_tokens[:, t - self.config.compression_window : t].cpu().numpy()
                )
                ratios_floats = [
                    self.compression_ratio(
                        np.array_str(cpu_tokens[i]).replace("\n", "")
                    )
                    for i in range(B * nbest)
                ]
                ratios = torch.tensor(ratios_floats, device=device)
                early_stopping_mask = torch.logical_and(
                    ratios > self.config.compression_threshold,
                    t > context_lengths + self.config.compression_window,
                )
                eos_mask = torch.logical_or(eos_mask, early_stopping_mask)

            # Decide if we are done
            done = bool(
                torch.logical_or(
                    torch.all(eos_mask),
                    t >= self.model.max_generation_length - 4,
                )
            )
            t += 1

        # Get final tokens, only use top hypo
        out_tokens = out_tokens[::nbest]
        valid_tokens_mask = torch.logical_and(
            torch.logical_and(
                out_tokens != self.pad_idx,
                out_tokens != self.bos_idx,  # type: ignore[arg-type]
            ),
            out_tokens != self.eos_idx,  # type: ignore[arg-type]
        )
        valid_tokens_count = valid_tokens_mask.sum(dim=1)
        final_tokens = torch.full(
            [B, int(valid_tokens_count.max())],
            fill_value=self.pad_idx,
            dtype=torch.int64,
            device=device,
        )
        for i in range(B):
            final_tokens[i, : valid_tokens_count[i]] = out_tokens[i][
                valid_tokens_mask[i]
            ]
        final_lengths = valid_tokens_count.tolist()

        return final_tokens, final_lengths
