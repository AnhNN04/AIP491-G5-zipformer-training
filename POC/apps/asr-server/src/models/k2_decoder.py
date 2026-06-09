"""
Real ASR inference engine using the Zipformer TorchScript JIT model.

Implements:
  - Model loading via torch.jit.load (jit_script.pt is the TorchScript export of epoch-12 weights)
  - Feature extraction via torchaudio.compliance.kaldi.fbank (Kaldi-compatible, no C++ kaldifeat)
  - Greedy search decoding (ported from ASR/zipformer/jit_pretrained.py)
  - Modified beam search decoding (batch mode)
  - Streaming decode_chunk for session-based WebSocket inference

Constitution compliance:
  - Principle V: No raw print() statements; all output via logging.
  - Principle II: decode/decode_chunk are synchronous CPU methods — caller must run in executor.
"""

import io
import logging
import math
from typing import Any, Dict, List, Optional

import k2
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger("asr-server.decoder")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_fbank(waveform_1d: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extract 80-dim Kaldi-compatible log-Mel Fbank features.

    Args:
        waveform_1d: 1-D float32 tensor of raw audio samples.
        sample_rate: Expected sample rate (must be 16000).

    Returns:
        2-D float32 tensor of shape (T, 80).
    """
    # torchaudio.compliance.kaldi.fbank expects (1, num_samples)
    wave_2d = waveform_1d.unsqueeze(0)
    features = torchaudio.compliance.kaldi.fbank(
        wave_2d,
        dither=0.0,
        snip_edges=False,
        sample_frequency=float(sample_rate),
        num_mel_bins=80,
        high_freq=7600.0,   # Nyquist (8000) - 400 matches kaldifeat default
        low_freq=20.0,
    )
    return features  # (T, 80)


def _audio_bytes_to_tensor(audio_bytes: bytes) -> torch.Tensor:
    """Convert raw WAV/audio bytes to a 1-D float32 waveform tensor at 16kHz.

    Uses soundfile as backend (no torchcodec dependency).

    Args:
        audio_bytes: Raw bytes of an audio file (WAV, FLAC, etc.).

    Returns:
        1-D float32 tensor of audio samples (16kHz, single channel).

    Raises:
        ValueError: If the audio sample rate is not 16kHz.
    """
    buf = io.BytesIO(audio_bytes)
    data, sample_rate = sf.read(buf, dtype="float32")
    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16kHz audio for ASR, but received {sample_rate}Hz. "
            "Pre-process audio to 16kHz mono before sending to this endpoint."
        )
    # If stereo, use only the first channel
    if data.ndim > 1:
        data = data[:, 0]
    return torch.tensor(data, dtype=torch.float32)


def _pcm_bytes_to_tensor(pcm_bytes: bytes) -> torch.Tensor:
    """Convert raw int16 PCM bytes to a 1-D float32 waveform tensor.

    Args:
        pcm_bytes: Raw int16 little-endian PCM bytes at 16kHz.

    Returns:
        1-D float32 tensor normalised to [-1.0, 1.0].
    """
    raw = torch.frombuffer(bytearray(pcm_bytes), dtype=torch.int16)
    return raw.float() / 32768.0


# ---------------------------------------------------------------------------
# Greedy search (ported from ASR/zipformer/jit_pretrained.py)
# ---------------------------------------------------------------------------

def _greedy_search_batch(
    model: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
) -> List[List[int]]:
    """Batch greedy search (max-sym-per-frame = 1).

    Args:
        model: The loaded TorchScript ASR model.
        encoder_out: (N, T, C) encoder output.
        encoder_out_lens: (N,) lengths.

    Returns:
        List of token-ID lists, one per utterance.
    """
    assert encoder_out.ndim == 3

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = encoder_out.device
    blank_id: int = model.decoder.blank_id
    context_size: int = model.decoder.context_size

    batch_size_list = packed.batch_sizes.tolist()
    N = encoder_out.size(0)

    hyps = [[blank_id] * context_size for _ in range(N)]
    decoder_input = torch.tensor(hyps, device=device, dtype=torch.int64)
    decoder_out = model.decoder(decoder_input, need_pad=torch.tensor([False])).squeeze(1)

    offset = 0
    for batch_size in batch_size_list:
        start, end = offset, offset + batch_size
        current_encoder_out = packed.data[start:end]
        offset = end

        decoder_out = decoder_out[:batch_size]
        logits = model.joiner(current_encoder_out, decoder_out)
        y_list = logits.argmax(dim=1).tolist()

        emitted = False
        for i, v in enumerate(y_list):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True

        if emitted:
            decoder_input = torch.tensor(
                [h[-context_size:] for h in hyps[:batch_size]],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input, need_pad=torch.tensor([False])
            ).squeeze(1)

    sorted_ans = [h[context_size:] for h in hyps]
    ans: List[List[int]] = []
    for i in range(N):
        ans.append(sorted_ans[packed.unsorted_indices.tolist()[i]])
    return ans


# ---------------------------------------------------------------------------
# Modified beam search (batch, ported from ASR/zipformer/beam_search.py logic)
# ---------------------------------------------------------------------------

def _modified_beam_search(
    model: torch.jit.ScriptModule,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam_size: int = 4,
) -> List[List[int]]:
    """Modified beam search — keep top-K hypotheses per frame.

    Args:
        model: TorchScript ASR model.
        encoder_out: (N, T, C).
        encoder_out_lens: (N,).
        beam_size: Number of active hypotheses to maintain.

    Returns:
        List of best token-ID lists, one per utterance.
    """
    device = encoder_out.device
    blank_id: int = model.decoder.blank_id
    context_size: int = model.decoder.context_size
    N = encoder_out.size(0)

    # For each utterance, maintain a list of (score, token_ids) tuples
    # Scores are log-probabilities
    beams: List[List[List]] = []
    for _ in range(N):
        initial = [[blank_id] * context_size]  # context
        beams.append([[0.0, initial[0]]])  # [[log_score, token_ids], ...]

    for t_step in range(encoder_out.size(1)):
        new_beams: List[List[List]] = [[] for _ in range(N)]

        for utt_idx in range(N):
            if t_step >= encoder_out_lens[utt_idx].item():
                new_beams[utt_idx] = beams[utt_idx]
                continue

            cur_encoder_frame = encoder_out[utt_idx, t_step:t_step+1]  # (1, C)
            active = beams[utt_idx]  # list of [log_score, token_ids]

            # Gather unique decoder contexts
            contexts = [h[1][-context_size:] for h in active]
            dec_input = torch.tensor(contexts, device=device, dtype=torch.int64)
            dec_out = model.decoder(dec_input, need_pad=torch.tensor([False])).squeeze(1)
            # dec_out: (num_hyps, D)

            # joiner expects (num_hyps, C) and (num_hyps, D)
            enc_repeated = cur_encoder_frame.expand(len(active), -1)
            logits = model.joiner(enc_repeated, dec_out)  # (num_hyps, vocab_size)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            candidates: List = []
            vocab_size = log_probs.size(-1)
            for h_idx, hyp in enumerate(active):
                log_score, token_ids = hyp
                for token_id in range(vocab_size):
                    new_score = log_score + log_probs[h_idx, token_id].item()
                    if token_id == blank_id:
                        candidates.append([new_score, token_ids])
                    else:
                        candidates.append([new_score, token_ids + [token_id]])

            # Prune to top beam_size
            candidates.sort(key=lambda x: x[0], reverse=True)
            new_beams[utt_idx] = candidates[:beam_size]

        beams = new_beams

    # Extract best hypothesis (highest score) per utterance
    results = []
    for utt_idx in range(N):
        best = max(beams[utt_idx], key=lambda x: x[0])
        # Remove the initial blank context tokens
        token_ids = best[1][context_size:]
        results.append(token_ids)
    return results


# ---------------------------------------------------------------------------
# K2Decoder — main serving class
# ---------------------------------------------------------------------------

class K2Decoder:
    """Real Zipformer ASR inference engine using TorchScript JIT model.

    Loads the model and token table once at startup. Provides synchronous
    decode() and decode_chunk() methods — callers should run these in a
    thread pool executor to avoid blocking the async event loop.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokens_path: Optional[str] = None,
    ) -> None:
        """Load model and symbol table at startup.

        Args:
            checkpoint_path: Path to jit_script.pt TorchScript checkpoint.
            tokens_path: Path to tokens.txt BPE symbol table.
        """
        self.checkpoint_path = checkpoint_path
        self.tokens_path = tokens_path

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            logger.info("CUDA available — using GPU for inference.")

        self.model: Optional[torch.jit.ScriptModule] = None
        self.token_table: Optional[k2.SymbolTable] = None

        if checkpoint_path:
            self._load_model(checkpoint_path)
        if tokens_path:
            self._load_tokens(tokens_path)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load TorchScript JIT model from disk."""
        logger.info(f"Loading TorchScript model from: {checkpoint_path}")
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        logger.info("TorchScript model loaded successfully.")

    def _load_tokens(self, tokens_path: str) -> None:
        """Load BPE symbol table from disk."""
        logger.info(f"Loading BPE symbol table from: {tokens_path}")
        self.token_table = k2.SymbolTable.from_file(tokens_path)
        logger.info(f"Symbol table loaded — vocab size: {len(self.token_table.symbols)}")

    def _token_ids_to_text(self, token_ids: List[int]) -> str:
        """Convert BPE token IDs to a clean Vietnamese text string."""
        if self.token_table is None:
            return ""
        text = ""
        for tid in token_ids:
            text += self.token_table[tid]
        return text.replace("▁", " ").strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self,
        audio_data: bytes,
        method: str = "greedy_search",
        beam_size: int = 4,
        causal: bool = False,
        chunk_size: Optional[int] = None,
        left_context_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Transcribe a WAV audio file (bytes) synchronously.

        Args:
            audio_data: Raw bytes of a WAV/FLAC audio file at 16kHz.
            method: 'greedy_search' or 'modified_beam_search'.
            beam_size: Beam width for modified_beam_search (1–20).
            causal: If True, use causal (online) decoding constraints.
            chunk_size: Streaming chunk size (informational only for batch mode).
            left_context_frames: Left context frames for streaming (informational).

        Returns:
            Dict with keys: text, confidence, word_alignments.

        Raises:
            ValueError: If method or beam_size are invalid, or model not loaded.
        """
        if self.model is None or self.token_table is None:
            raise ValueError(
                "Model or token table not loaded. "
                "Ensure ASR_CHECKPOINT_PATH and ASR_TOKENS_PATH are configured correctly."
            )

        if method not in ("greedy_search", "modified_beam_search"):
            raise ValueError(f"Unsupported decoding method: '{method}'")

        if not (1 <= beam_size <= 20):
            raise ValueError(f"beam_size must be between 1 and 20, got: {beam_size}")

        logger.info(f"decode() → method={method}, beam_size={beam_size}, causal={causal}")

        # 1. Audio bytes → float32 waveform tensor
        waveform = _audio_bytes_to_tensor(audio_data)
        waveform = waveform.to(self.device)

        # 2. Extract Fbank features
        features = _extract_fbank(waveform)  # (T, 80)
        feature_lengths = torch.tensor([features.size(0)], dtype=torch.int32, device=self.device)

        # 3. Batch the single utterance
        features_batch = features.unsqueeze(0).to(self.device)  # (1, T, 80)

        # 4. Encoder forward
        with torch.no_grad():
            encoder_out, encoder_out_lens = self.model.encoder(
                features=features_batch,
                feature_lengths=feature_lengths,
            )

        # 5. Decoder/search
        with torch.no_grad():
            if method == "greedy_search":
                hyps = _greedy_search_batch(self.model, encoder_out, encoder_out_lens)
            else:
                hyps = _modified_beam_search(
                    self.model, encoder_out, encoder_out_lens, beam_size=beam_size
                )

        # 6. Token IDs → text
        text = self._token_ids_to_text(hyps[0])
        confidence = 0.95  # Placeholder; real confidence needs lattice scores

        logger.info(f"Decoded transcript ({len(hyps[0])} tokens): {text[:80]}...")

        return {
            "text": text,
            "confidence": confidence,
            "word_alignments": [],  # Alignment extraction requires a separate pass
        }

    def decode_chunk(
        self,
        accumulated_samples: torch.Tensor,
        method: str = "greedy_search",
        beam_size: int = 4,
    ) -> str:
        """Transcribe accumulated waveform samples (for WebSocket streaming).

        The non-causal model processes the entire accumulated buffer every call.
        Callers should accumulate samples and call this for each incoming chunk.

        Args:
            accumulated_samples: 1-D float32 tensor of all received samples so far.
            method: 'greedy_search' or 'modified_beam_search'.
            beam_size: Beam width for modified_beam_search.

        Returns:
            Current full transcript string.
        """
        if self.model is None or self.token_table is None:
            return ""

        if accumulated_samples.numel() < 800:  # ~0.05s minimum
            return ""

        waveform = accumulated_samples.to(self.device)
        features = _extract_fbank(waveform)  # (T, 80)
        feature_lengths = torch.tensor([features.size(0)], dtype=torch.int32, device=self.device)
        features_batch = features.unsqueeze(0)

        with torch.no_grad():
            encoder_out, encoder_out_lens = self.model.encoder(
                features=features_batch,
                feature_lengths=feature_lengths,
            )

        with torch.no_grad():
            if method == "greedy_search":
                hyps = _greedy_search_batch(self.model, encoder_out, encoder_out_lens)
            else:
                hyps = _modified_beam_search(
                    self.model, encoder_out, encoder_out_lens, beam_size=beam_size
                )

        return self._token_ids_to_text(hyps[0])

    def decode_stream_session(
        self,
        stream: "DecodeStream",  # type: ignore[name-defined]  # avoid circular import
    ) -> str:
        """Transcribe all accumulated audio in a DecodeStream session object.

        This is the primary entry point for WebSocket streaming — it reads
        the session's accumulated buffer and decoding configuration in one call.

        Args:
            stream: A DecodeStream instance with accumulated PCM samples and
                    session-level config (method, beam_size).

        Returns:
            Current full transcript string for the session.
        """
        return self.decode_chunk(
            accumulated_samples=stream.get_samples(),
            method=stream.method,
            beam_size=stream.beam_size,
        )
