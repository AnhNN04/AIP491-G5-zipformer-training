import os
import asyncio
import logging
import tempfile
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class InvalidAudioFormatException(Exception):
    """Raised when the uploaded file format is not supported."""
    pass

class TranscodingError(Exception):
    """Raised when the FFmpeg transcoding process fails."""
    pass

class AudioProcessor:
    SUPPORTED_FORMATS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'mp4'}

    def __init__(self, tmp_dir: Optional[str] = None):
        # Locate or create a tmp directory in the web-backend app directory
        if tmp_dir:
            self.tmp_dir = tmp_dir
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.tmp_dir = os.path.join(base_dir, "tmp")
            
        os.makedirs(self.tmp_dir, exist_ok=True)

    def validate_format(self, original_format: str) -> None:
        """Verify if the file extension is supported."""
        fmt = original_format.lower().replace(".", "")
        if fmt not in self.SUPPORTED_FORMATS:
            raise InvalidAudioFormatException(
                f"Unsupported format: {original_format}. Supported formats: {list(self.SUPPORTED_FORMATS)}"
            )

    async def get_duration(self, file_path: str) -> float:
        """
        Use ffprobe to asynchronously extract audio duration in seconds.
        """
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise TranscodingError(f"ffprobe failed to extract duration: {stderr.decode().strip()}")
            
            output_str = stdout.decode().strip()
            if not output_str:
                raise TranscodingError("ffprobe returned empty duration output.")
                
            return float(output_str)
        except Exception as e:
            logger.error(f"Error executing ffprobe: {e}")
            raise TranscodingError(f"Failed to read audio file metadata: {e}") from e

    async def transcode(self, audio_data: bytes, original_format: str) -> Tuple[bytes, float]:
        """
        Transcodes input audio bytes to standard 16kHz mono 16-bit PCM WAV.
        Returns a tuple: (transcoded_wav_bytes, duration_seconds).
        """
        self.validate_format(original_format)
        
        suffix = f".{original_format.lower().replace('.', '')}"
        
        # Create temp files within the workspace tmp directory
        in_fd, in_path = tempfile.mkstemp(dir=self.tmp_dir, suffix=suffix)
        out_fd, out_path = tempfile.mkstemp(dir=self.tmp_dir, suffix=".wav")
        
        # Close file descriptors immediately as we will write/read using file paths
        os.close(in_fd)
        os.close(out_fd)
        
        try:
            # Write input bytes to file
            with open(in_path, "wb") as f:
                f.write(audio_data)

            # Spawn ffmpeg subprocess asynchronously
            # Options: 16kHz (-ar 16000), mono (-ac 1), 16-bit PCM WAV (-c:a pcm_s16le)
            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path
            ]
            
            logger.info(f"Spawning ffmpeg transcode: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                err_msg = stderr.decode().strip()
                logger.error(f"ffmpeg transcoding failed: {err_msg}")
                raise TranscodingError(f"Transcoding failed: {err_msg}")

            # Get duration of standard WAV file
            duration = await self.get_duration(out_path)
            
            # Read standardized audio bytes
            with open(out_path, "rb") as f:
                transcoded_bytes = f.read()

            logger.info(
                f"Successfully transcoded file from {original_format} to WAV. Duration: {duration:.2f}s"
            )
            return transcoded_bytes, duration

        finally:
            # Always clean up temporary files
            for path in (in_path, out_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")
