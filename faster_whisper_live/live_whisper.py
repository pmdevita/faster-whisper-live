import asyncio
import concurrent.futures
import math
import typing

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from .constants import SAMPLE_RATE
from .ffmpeg import FFmpegDecoder, AsyncFFmpegDecoder


def bytes_length_to_audio_length(data):
    return len(data) / SAMPLE_RATE / 2


def audio_length_to_bytes_length(seconds):
    return math.floor(seconds * SAMPLE_RATE) * 2


def get_segments(segments):
    return list(segments)


class LiveWhisper:
    def __init__(
            self,
            model_size_or_path: str,
            device: str = "auto",
            device_index: typing.Union[int, typing.List[int]] = 0,
            compute_type: str = "default",
            cpu_threads: int = 0,
            num_workers: int = 1,
            download_root: typing.Optional[str] = None,
            local_files_only: bool = False,
    ):
        self.model = WhisperModel(model_size_or_path, device=device, device_index=device_index,
                                  compute_type=compute_type, cpu_threads=cpu_threads, num_workers=num_workers,
                                  download_root=download_root, local_files_only=local_files_only)

    def transcribe(self, file: typing.BinaryIO, default_interval=5, chunk_margin=2, decode_audio=True,
                   ignore_eof=False, ffmpeg="ffmpeg", **kwargs) -> typing.Generator[Segment, None, None]:
        """
        Transcribe an audio file. Takes a file-like object, a few settings for controlling the live
        transcription, and then the normal keyword arguments for WhisperModel.transcode()
        :param ffmpeg: Path to the FFmpeg binary
        :param ignore_eof: Whether transcription should continue even if end of file is reached (defaults to False)
        :param decode_audio: Whether LiveWhisper should decode the audio with FFmpeg (defaults to True)
        :param chunk_margin: How much time should a segment have after it (used to determine if an audio chunk fully
        held a speech segment)
        :param file: A file-like object that contains properly formatted PCM data
        :param default_interval: How many seconds to transcribe at once
        :return:
        """
        f = file
        if decode_audio:
            f = FFmpegDecoder(file, ffmpeg=ffmpeg)

        n_bytes = default_interval * SAMPLE_RATE * 2

        chunk = f.read(n_bytes)

        # Have we reached the end of the file?
        # If we have, allow text transcription to run to the end of the chunk
        end_of_file = False

        try:
            while chunk:
                # Determine length of chunk in seconds
                chunk_length = bytes_length_to_audio_length(chunk)

                # Convert audio from 16-bit ints to 32-bit floats
                audio = np.frombuffer(chunk, np.int16).flatten().astype(np.float32) / 32768.0

                # How many seconds should we chop off of this audio chunk after this transcription run?
                # We remove audio once we've determined that segment has been fully processed
                new_start = 0

                # Perform the transcription
                segments, audio_info = self.model.transcribe(audio, **kwargs)
                for segment in segments:
                    # Once we reach segments that have their ends within the chunk margin, break out
                    # Unless we're at the end of the file, in which case we have no choice
                    if segment.end > chunk_length - chunk_margin and not end_of_file:
                        break
                    yield segment
                    new_start = segment.end

                # Remove the beginning
                chunk = chunk[audio_length_to_bytes_length(new_start):]

                # Load the next chunk
                new_chunk = f.read(n_bytes)

                # If we got blank, we're at the end of the file.
                # If we aren't ignoring EOF, set the flag
                if not new_chunk and not ignore_eof:
                    end_of_file = True

                # Append the newly chunked data to our current chunk
                chunk += new_chunk
        except KeyboardInterrupt:
            pass
        finally:
            print("goodbye")
            if decode_audio:
                f.terminate()


class AsyncLiveWhisper(LiveWhisper):
    THREAD_POOL = concurrent.futures.ThreadPoolExecutor()

    async def transcribe(self, file: asyncio.StreamReader, default_interval=5, chunk_margin=2, decode_audio=True,
                          ignore_eof=False, ffmpeg="ffmpeg", **kwargs) -> typing.Generator[Segment, None, None]:
        """
        Transcribe an audio file asynchronously. Takes a file-like object, a few settings for controlling the live
        transcription, and then the normal keyword arguments for WhisperModel.transcode()
        :param ffmpeg: Path to the FFmpeg binary
        :param ignore_eof: Whether transcription should continue even if end of file is reached (defaults to False)
        :param decode_audio: Whether LiveWhisper should decode the audio with FFmpeg (defaults to True)
        :param chunk_margin: How much time should a segment have after it (used to determine if an audio chunk fully
        held a speech segment)
        :param file: A file-like object that contains properly formatted PCM data
        :param default_interval: How many seconds to transcribe at once
        :return:
        """
        f = file
        if decode_audio:
            f = AsyncFFmpegDecoder(file, ffmpeg=ffmpeg)
            await f.start()

        n_bytes = default_interval * SAMPLE_RATE * 2

        chunk = await f.read(n_bytes)

        # Have we reached the end of the file?
        # If we have, allow text transcription to run to the end of the
        print('asdf')
        end_of_file = False

        try:
            while chunk:
                # Determine length of chunk in seconds
                chunk_length = bytes_length_to_audio_length(chunk)

                # Convert audio from 16-bit ints to 32-bit floats
                audio = np.frombuffer(chunk, np.int16).flatten().astype(np.float32) / 32768.0

                # How many seconds should we chop off of this audio chunk after this transcription run?
                # We remove audio once we've determined that segment has been fully processed
                new_start = 0

                # Perform the transcription
                segments, audio_info = self.model.transcribe(audio, **kwargs)

                segments = await asyncio.get_running_loop().run_in_executor(self.THREAD_POOL, get_segments, segments)
                print(segments)
                for segment in segments:
                    # Once we reach segments that have their ends within the chunk margin, break out
                    # Unless we're at the end of the file, in which case we have no choice
                    if segment.end > chunk_length - chunk_margin and not end_of_file:
                        break
                    yield segment
                    new_start = segment.end

                # Remove the beginning
                chunk = chunk[audio_length_to_bytes_length(new_start):]

                # Load the next chunk
                new_chunk = await f.read(n_bytes)

                # If we got blank, we're at the end of the file.
                # If we aren't ignoring EOF, set the flag
                if not new_chunk and not ignore_eof:
                    end_of_file = True

                # Append the newly chunked data to our current chunk
                chunk += new_chunk
        except KeyboardInterrupt:
            pass
        finally:
            if decode_audio:
                f.terminate()


