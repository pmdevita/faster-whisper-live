# faster-whisper-live

Use faster-whisper with a streaming audio source. Includes support for asyncio.

Special thanks to [JonathanFly](https://github.com/JonathanFly) for his 
initial implementation [here](https://github.com/JonathanFly/faster-whisper-livestream-translator).

This is still a work in progress, might break sometimes. Contributions welcome and appreciated!

## Installation

I'd like to get this in a better state before uploading to PyPI. For now you can do

```pip install pip install git+https://github.com/pmdevita/faster-whisper-live.git```

or if you're using Poetry

```poetry add git+https://github.com/pmdevita/faster-whisper-live.git```

## Usage

LiveWhisper takes the same arguments for initialization and transcription as LiveWhisper does.
However, you can pass it a file-like object, and it will stream output from it.

```python
from faster_whisper_live import LiveWhisper

model = LiveWhisper("small", compute_type="int8_float16")
with open("whispertest.m4a", "rb") as f:
    for segment in model.transcribe(f, vad_filter=True):
        print(segment)
```

Asyncio is also supported.

```python
import asyncio
import aiofiles
from faster_whisper_live import AsyncLiveWhisper


async def main():
    model = AsyncLiveWhisper("small", compute_type="int8_float16")
    async with aiofiles.open("whispertest.m4a", "rb") as f:
        async for segment in model.transcribe(f, vad_filter=True):
            print(segment)


asyncio.run(main())
```


## Advanced decoding for faster-whisper

If you'd like to supply your own PCM audio stream, it needs to be:

- Sample Rate: 16kHz
- Mono audio
- 16-bit Little-endian PCM (s16le)

You then need to pass `decode_audio=False` to `transcribe`.

An example ffmpeg command to generate such a file would be:

```shell
ffmpeg -i input_file.m4a -f s16le -ac 1 -ar 16000 output.pcm
```


