"""
Unit tests for DecodeStream: session state management,
PCM accumulation, config extraction, and reset behaviour.

Run:
    PYTHONPATH=. pytest tests/test_decode_stream.py -v
"""

import pytest
import torch
import numpy as np
from src.models.decode_stream import DecodeStream


def _make_pcm_bytes(n_samples: int, value_int16: int = 0) -> bytes:
    """Generate raw int16 PCM bytes."""
    pcm = np.full(n_samples, value_int16, dtype=np.int16)
    return pcm.tobytes()


class TestDecodeStreamInit:
    def test_default_config(self):
        """Default config should use greedy_search, beam_size=4."""
        stream = DecodeStream(session_id="test_sess_001")
        assert stream.method == "greedy_search"
        assert stream.beam_size == 4
        assert stream.causal is False
        assert stream.chunk_size is None
        assert stream.left_context_frames is None
        assert stream.num_chunks == 0
        assert stream.duration_seconds == pytest.approx(0.0)

    def test_custom_config_extraction(self):
        """All config keys should be correctly parsed."""
        stream = DecodeStream(
            session_id="test_sess_002",
            config={
                "method": "modified_beam_search",
                "beam_size": 8,
                "causal": True,
                "chunk_size": 32,
                "left_context_frames": 256,
            }
        )
        assert stream.method == "modified_beam_search"
        assert stream.beam_size == 8
        assert stream.causal is True
        assert stream.chunk_size == 32
        assert stream.left_context_frames == 256

    def test_session_id_stored(self):
        """Session ID should be accessible."""
        stream = DecodeStream(session_id="unique_session_xyz")
        assert stream.session_id == "unique_session_xyz"

    def test_none_config_uses_defaults(self):
        """Passing None for config should use all defaults."""
        stream = DecodeStream(session_id="s1", config=None)
        assert stream.method == "greedy_search"


class TestDecodeStreamAddChunk:
    def test_single_chunk_accumulated(self):
        """Single PCM chunk should accumulate correctly."""
        stream = DecodeStream(session_id="s1")
        pcm = _make_pcm_bytes(1600)  # 0.1s at 16kHz
        stream.add_chunk(pcm)
        assert stream.num_chunks == 1
        assert stream.duration_seconds == pytest.approx(0.1, abs=0.001)
        samples = stream.get_samples()
        assert samples.numel() == 1600
        assert samples.dtype == torch.float32

    def test_multiple_chunks_accumulated(self):
        """Multiple PCM chunks should stack into one buffer."""
        stream = DecodeStream(session_id="s2")
        pcm = _make_pcm_bytes(800)
        stream.add_chunk(pcm)
        stream.add_chunk(pcm)
        stream.add_chunk(pcm)
        assert stream.num_chunks == 3
        assert stream.get_samples().numel() == 2400

    def test_pcm_normalisation(self):
        """int16 max (32767) should map to ~1.0 float."""
        stream = DecodeStream(session_id="s3")
        pcm = np.full(100, 32767, dtype=np.int16).tobytes()
        stream.add_chunk(pcm)
        samples = stream.get_samples()
        assert samples.abs().max().item() == pytest.approx(32767 / 32768.0, rel=1e-4)

    def test_zero_pcm_maps_to_zero_float(self):
        """Zero PCM bytes should produce a zero float tensor."""
        stream = DecodeStream(session_id="s4")
        pcm = _make_pcm_bytes(400, value_int16=0)
        stream.add_chunk(pcm)
        samples = stream.get_samples()
        assert samples.abs().max().item() == pytest.approx(0.0)


class TestDecodeStreamGetSamples:
    def test_get_samples_returns_copy(self):
        """get_samples() should return a copy, not a view."""
        stream = DecodeStream(session_id="s5")
        stream.add_chunk(_make_pcm_bytes(400))
        samples_a = stream.get_samples()
        samples_b = stream.get_samples()
        # Mutating one copy should not affect the stream
        samples_a.fill_(999.0)
        samples_c = stream.get_samples()
        assert not torch.all(samples_c == 999.0)
        assert torch.allclose(samples_b, samples_c)

    def test_get_samples_empty_before_any_chunk(self):
        """No chunks → empty tensor."""
        stream = DecodeStream(session_id="s6")
        samples = stream.get_samples()
        assert samples.numel() == 0


class TestDecodeStreamReset:
    def test_reset_clears_buffer(self):
        """reset() should clear accumulated samples and chunk count."""
        stream = DecodeStream(session_id="s7")
        stream.add_chunk(_make_pcm_bytes(3200))
        assert stream.num_chunks == 1
        stream.reset()
        assert stream.num_chunks == 0
        assert stream.get_samples().numel() == 0
        assert stream.duration_seconds == pytest.approx(0.0)

    def test_reset_preserves_config(self):
        """reset() should NOT change the session config."""
        stream = DecodeStream(
            session_id="s8",
            config={"method": "modified_beam_search", "beam_size": 12}
        )
        stream.reset()
        assert stream.method == "modified_beam_search"
        assert stream.beam_size == 12

    def test_can_accumulate_after_reset(self):
        """After reset, new chunks should accumulate correctly."""
        stream = DecodeStream(session_id="s9")
        stream.add_chunk(_make_pcm_bytes(1600))
        stream.reset()
        stream.add_chunk(_make_pcm_bytes(800))
        assert stream.get_samples().numel() == 800


class TestDecodeStreamRepr:
    def test_repr_contains_session_id(self):
        stream = DecodeStream(session_id="test_repr_001")
        assert "test_repr_001" in repr(stream)
