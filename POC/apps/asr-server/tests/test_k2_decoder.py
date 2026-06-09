"""
Unit tests for K2Decoder: model loading, parameter validation,
feature extraction, and decoding (mocked and integration).

Tests are designed to run:
  - Without model files: all parameter validation and helper function tests.
  - With model files (INTEGRATION): integration tests gated on CHECKPOINT_PATH.

Run:
    uv run pytest tests/test_k2_decoder.py -v
    uv run pytest tests/test_k2_decoder.py -v -m integration  # integration only
"""

import io
import os
import struct
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

import soundfile as sf

from src.models.k2_decoder import (
    K2Decoder,
    _extract_fbank,
    _audio_bytes_to_tensor,
    _pcm_bytes_to_tensor,
)


# ---------------------------------------------------------------------------
# Helpers for generating test audio
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate silent WAV bytes at the given sample rate."""
    n_samples = int(duration_s * sample_rate)
    samples = np.zeros(n_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


def _make_pcm_bytes(n_samples: int = 3200) -> bytes:
    """Generate raw int16 PCM bytes (silence, 0.2s at 16kHz)."""
    pcm = np.zeros(n_samples, dtype=np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Feature extraction tests (no model required)
# ---------------------------------------------------------------------------

class TestExtractFbank:
    def test_output_shape(self):
        """Fbank should output (T, 80) for any length waveform."""
        waveform = torch.zeros(16000)  # 1 second silence
        feats = _extract_fbank(waveform, sample_rate=16000)
        assert feats.ndim == 2
        assert feats.shape[1] == 80
        assert feats.shape[0] > 0

    def test_short_audio(self):
        """Short audio (0.05s) should still produce valid features."""
        waveform = torch.zeros(800)  # 50ms
        feats = _extract_fbank(waveform)
        assert feats.ndim == 2
        assert feats.shape[1] == 80


class TestAudioBytesToTensor:
    def test_valid_16khz_wav(self):
        """Valid 16kHz WAV should produce a 1-D float32 tensor."""
        wav_bytes = _make_wav_bytes(0.5, 16000)
        tensor = _audio_bytes_to_tensor(wav_bytes)
        assert tensor.ndim == 1
        assert tensor.dtype == torch.float32
        assert tensor.shape[0] == 8000  # 0.5s * 16000

    def test_wrong_sample_rate_raises(self):
        """Audio at 8kHz should raise ValueError."""
        n_samples = 4000
        samples = np.zeros(n_samples, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, samples, 8000, format="WAV", subtype="FLOAT")
        wav_bytes = buf.getvalue()
        with pytest.raises(ValueError, match="Expected 16kHz"):
            _audio_bytes_to_tensor(wav_bytes)

    def test_stereo_uses_first_channel(self):
        """Stereo audio should be downmixed to mono using first channel."""
        n_samples = 8000
        samples = np.ones((n_samples, 2), dtype=np.float32)
        samples[:, 1] = 0.0  # channel 2 is silence
        buf = io.BytesIO()
        sf.write(buf, samples, 16000, format="WAV", subtype="FLOAT")
        wav_bytes = buf.getvalue()
        tensor = _audio_bytes_to_tensor(wav_bytes)
        assert tensor.ndim == 1
        assert tensor.shape[0] == n_samples
        assert tensor.mean().item() == pytest.approx(1.0, abs=1e-5)


class TestPcmBytesToTensor:
    def test_valid_pcm(self):
        """Raw int16 PCM bytes should produce a float32 tensor in [-1, 1]."""
        pcm = _make_pcm_bytes(3200)
        tensor = _pcm_bytes_to_tensor(pcm)
        assert tensor.ndim == 1
        assert tensor.dtype == torch.float32
        assert tensor.shape[0] == 3200
        assert tensor.abs().max().item() <= 1.0

    def test_max_amplitude(self):
        """int16 max should map to 1.0."""
        pcm = np.array([32767], dtype=np.int16).tobytes()
        tensor = _pcm_bytes_to_tensor(pcm)
        assert tensor[0].item() == pytest.approx(32767 / 32768.0, rel=1e-5)


# ---------------------------------------------------------------------------
# K2Decoder parameter validation (no model required)
# ---------------------------------------------------------------------------

class TestK2DecoderValidation:
    def test_decode_raises_when_model_not_loaded(self):
        """decode() must raise ValueError if model/tokens not loaded."""
        decoder = K2Decoder()  # No paths → model=None, token_table=None
        with pytest.raises(ValueError, match="Model or token table not loaded"):
            decoder.decode(_make_wav_bytes())

    def test_invalid_method_raises(self):
        """Unsupported decoding method should raise ValueError."""
        decoder = K2Decoder()
        # Bypass model check by monkey-patching
        decoder.model = MagicMock()
        decoder.token_table = MagicMock()
        decoder.token_table.__len__ = MagicMock(return_value=2000)
        with pytest.raises(ValueError, match="Unsupported decoding method"):
            decoder.decode(_make_wav_bytes(), method="invalid_search")

    def test_invalid_beam_size_too_small_raises(self):
        """beam_size < 1 should raise ValueError."""
        decoder = K2Decoder()
        decoder.model = MagicMock()
        decoder.token_table = MagicMock()
        with pytest.raises(ValueError, match="beam_size must be between"):
            decoder.decode(_make_wav_bytes(), method="modified_beam_search", beam_size=0)

    def test_invalid_beam_size_too_large_raises(self):
        """beam_size > 20 should raise ValueError."""
        decoder = K2Decoder()
        decoder.model = MagicMock()
        decoder.token_table = MagicMock()
        with pytest.raises(ValueError, match="beam_size must be between"):
            decoder.decode(_make_wav_bytes(), method="modified_beam_search", beam_size=21)

    def test_decode_chunk_returns_empty_when_model_not_loaded(self):
        """decode_chunk() should return empty string if model not loaded."""
        decoder = K2Decoder()
        result = decoder.decode_chunk(torch.zeros(16000))
        assert result == ""

    def test_decode_chunk_returns_empty_for_very_short_audio(self):
        """decode_chunk() should skip if fewer than 800 samples."""
        decoder = K2Decoder()
        decoder.model = MagicMock()
        decoder.token_table = MagicMock()
        result = decoder.decode_chunk(torch.zeros(400))
        assert result == ""


# ---------------------------------------------------------------------------
# Integration tests: require real checkpoint and token table
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.getenv("ASR_CHECKPOINT_PATH", "")
TOKENS_PATH = os.getenv("ASR_TOKENS_PATH", "")
REAL_AUDIO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "input-test", "video-test-001-small.wav"
)

_model_available = bool(
    CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH)
    and TOKENS_PATH and os.path.exists(TOKENS_PATH)
)

requires_model = pytest.mark.skipif(
    not _model_available,
    reason="ASR_CHECKPOINT_PATH and ASR_TOKENS_PATH must be set to real model files."
)


@requires_model
class TestK2DecoderIntegration:
    @pytest.fixture(scope="class")
    def decoder(self):
        return K2Decoder(checkpoint_path=CHECKPOINT_PATH, tokens_path=TOKENS_PATH)

    def test_model_loaded(self, decoder):
        assert decoder.model is not None
        assert decoder.token_table is not None

    def test_greedy_search_returns_text(self, decoder):
        """Greedy search on synthetic silence should return a string."""
        wav_bytes = _make_wav_bytes(1.0, 16000)
        result = decoder.decode(wav_bytes, method="greedy_search")
        assert isinstance(result["text"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["word_alignments"], list)

    def test_modified_beam_search_returns_text(self, decoder):
        """Modified beam search on synthetic silence should return a string."""
        wav_bytes = _make_wav_bytes(1.0, 16000)
        result = decoder.decode(wav_bytes, method="modified_beam_search", beam_size=4)
        assert isinstance(result["text"], str)
        assert isinstance(result["confidence"], float)

    def test_real_audio_greedy_search(self, decoder):
        """Real audio file should produce non-empty Vietnamese transcript."""
        if not os.path.exists(REAL_AUDIO_PATH):
            pytest.skip("Real test audio file not found at expected path.")
        with open(REAL_AUDIO_PATH, "rb") as f:
            audio_bytes = f.read()
        result = decoder.decode(audio_bytes, method="greedy_search")
        assert len(result["text"]) > 0, "Transcript should be non-empty for real audio."

    def test_decode_chunk_real(self, decoder):
        """decode_chunk should return a string for real-length audio tensor."""
        samples = torch.zeros(16000)  # 1s silence
        text = decoder.decode_chunk(samples, method="greedy_search")
        assert isinstance(text, str)
