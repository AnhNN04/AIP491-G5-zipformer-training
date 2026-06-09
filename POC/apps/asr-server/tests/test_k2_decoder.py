import pytest
from src.models.k2_decoder import K2Decoder

def test_k2_decoder_greedy():
    decoder = K2Decoder()
    result = decoder.decode(b"audio", method="greedy_search")
    assert result["text"] == "chào mừng bạn đến với hệ thống nhận dạng giọng nói"
    assert len(result["word_alignments"]) > 0

def test_k2_decoder_beam():
    decoder = K2Decoder()
    result = decoder.decode(b"audio", method="modified_beam_search", beam_size=12, causal=True)
    assert result["text"] == "chào mừng bạn đến với hệ thống nhận dạng giọng nói"

def test_k2_decoder_invalid_method():
    decoder = K2Decoder()
    with pytest.raises(ValueError) as exc_info:
        decoder.decode(b"audio", method="invalid_search")
    assert "Unsupported decoding method" in str(exc_info.value)

def test_k2_decoder_invalid_beam_size():
    decoder = K2Decoder()
    with pytest.raises(ValueError) as exc_info:
        decoder.decode(b"audio", method="modified_beam_search", beam_size=0)
    assert "Beam size must be between" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        decoder.decode(b"audio", method="modified_beam_search", beam_size=21)
    assert "Beam size must be between" in str(exc_info.value)
