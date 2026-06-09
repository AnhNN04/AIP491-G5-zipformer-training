from uuid import uuid4
from src.domain.entities import User, AudioFile, DecodingConfig, Transcription, DialectStats

def test_entities_initialization():
    user = User(username="test_user", password_hash="hashed_pw")
    assert user.username == "test_user"
    assert user.password_hash == "hashed_pw"
    assert user.id is None
    
    audio_file_id = uuid4()
    config_id = uuid4()
    
    transcription = Transcription(
        audio_file_id=audio_file_id,
        decoding_config_id=config_id,
        text="xin chào",
        confidence=0.98,
        word_alignments=[{"word": "xin", "start": 0.0, "end": 0.5, "conf": 0.99}]
    )
    assert transcription.text == "xin chào"
    assert len(transcription.word_alignments) == 1
