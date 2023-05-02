import asyncio
import time

import aiofiles
from faster_whisper_live import AsyncLiveWhisper


async def main():
    model = AsyncLiveWhisper("large-v2", device="cuda", compute_type="int8_float16")

    async with aiofiles.open("whispertest.m4a", "rb") as f:
        temp_output_length = 0
        async for segment in model.transcribe(f, vad_filter=True):
            if segment.id == 1:
                if temp_output_length > 0:
                    time.sleep(2)
                print("".join(["\b" for i in range(temp_output_length)]), end="")
                temp_output_length = 0
            print(segment.text, end="")
            if segment.partial:
                temp_output_length += len(segment.text) + 1




asyncio.run(main())

