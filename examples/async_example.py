import asyncio
import aiofiles
from faster_whisper_live import AsyncLiveWhisper


async def main():
    model = AsyncLiveWhisper("small", compute_type="int8_float16")
    async with aiofiles.open("whispertest.m4a", "rb") as f:
        async for segment in model.transcribe(f, vad_filter=True):
            print(segment)


asyncio.run(main())

