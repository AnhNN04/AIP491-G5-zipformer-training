"""
DecodeStream: Session-level state manager for WebSocket streaming ASR.

Encapsulates the accumulated audio buffer and decoding configuration
for a single WebSocket session. The K2Decoder uses DecodeStream objects
to abstract session state away from the HTTP transport layer.

Constitution compliance:
  - Principle V: No raw print(); all output via logging.
  - Principle II: add_chunk() and get_samples() are synchronous — callers
    should run decode operations in a thread pool executor.
"""

import logging
from typing import Optional

import torch

from src.models.k2_decoder import _pcm_bytes_to_tensor

logger = logging.getLogger("asr-server.decode_stream")


class DecodeStream:
    """Per-session audio accumulation buffer for WebSocket streaming ASR.

    Responsibilities:
      - Accumulate raw int16 PCM chunks arriving from the WebSocket client.
      - Expose the accumulated float32 waveform tensor for the decoder.
      - Carry session-level decoding configuration (method, beam_size, etc.).

    Usage:
        stream = DecodeStream(session_id="asr_sess_abc123", config={...})
        stream.add_chunk(raw_pcm_bytes)      # Each WS binary frame
        samples = stream.get_samples()       # Pass to K2Decoder.decode_chunk()
        stream.reset()                       # After stop event
    """

    def __init__(self, session_id: str, config: Optional[dict] = None) -> None:
        """Initialise a new streaming session buffer.

        Args:
            session_id: Unique identifier for this WebSocket session.
            config: Dict with optional keys:
                - method: 'greedy_search' | 'modified_beam_search' (default: 'greedy_search')
                - beam_size: int 1-20 (default: 4)
                - causal: bool (default: False)
                - chunk_size: int | None (informational)
                - left_context_frames: int | None (informational)
        """
        self.session_id = session_id
        config = config or {}

        self.method: str = config.get("method", "greedy_search")
        self.beam_size: int = config.get("beam_size", 4)
        self.causal: bool = config.get("causal", False)
        self.chunk_size: Optional[int] = config.get("chunk_size")
        self.left_context_frames: Optional[int] = config.get("left_context_frames")

        # Accumulated waveform samples (float32, 16kHz, mono)
        self._samples: torch.Tensor = torch.tensor([], dtype=torch.float32)
        self._num_chunks: int = 0

        logger.debug(
            f"DecodeStream {session_id} created — method={self.method}, beam_size={self.beam_size}"
        )

    def add_chunk(self, pcm_bytes: bytes) -> None:
        """Append raw int16 PCM bytes to the accumulated buffer.

        Args:
            pcm_bytes: Raw int16 little-endian PCM bytes at 16kHz.
        """
        new_samples = _pcm_bytes_to_tensor(pcm_bytes)
        self._samples = torch.cat([self._samples, new_samples])
        self._num_chunks += 1
        logger.debug(
            f"DecodeStream {self.session_id}: chunk #{self._num_chunks} "
            f"(+{len(new_samples)} samples, total={self._samples.numel()})"
        )

    def get_samples(self) -> torch.Tensor:
        """Return a copy of the accumulated waveform tensor.

        Returns:
            1-D float32 tensor of all accumulated samples.
        """
        return self._samples.clone()

    def reset(self) -> None:
        """Clear the accumulated audio buffer (e.g. after stop event).

        Retains session configuration for potential reuse.
        """
        total_samples = self._samples.numel()
        self._samples = torch.tensor([], dtype=torch.float32)
        self._num_chunks = 0
        logger.info(
            f"DecodeStream {self.session_id} reset after {total_samples} samples."
        )

    @property
    def duration_seconds(self) -> float:
        """Duration of the accumulated audio in seconds (at 16kHz)."""
        return self._samples.numel() / 16000.0

    @property
    def num_chunks(self) -> int:
        """Number of PCM chunks added so far."""
        return self._num_chunks

    def __repr__(self) -> str:
        return (
            f"DecodeStream(session_id={self.session_id!r}, "
            f"method={self.method!r}, beam_size={self.beam_size}, "
            f"samples={self._samples.numel()}, chunks={self._num_chunks})"
        )
