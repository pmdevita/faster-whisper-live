# faster-whisper-live

Use faster-whisper with a streaming audio source. Includes support for asyncio.

Special thanks to @JonathanFly for his initial implementation [here](https://github.com/JonathanFly/faster-whisper-livestream-translator).

This is still a work in progress, but you can see examples in the `examples/` folder.


## Advanced decoding for faster-whisper

If you'd like to supply your own PCM audio stream, it needs to be:

- Sample Rate: 16kHz
- Mono audio
- 16-bit Little-endian PCM (s16le)

An example ffmpeg command to generate such a file would be:

```shell
ffmpeg -i input_file.m4a -f s16le -ac 1 -ar 16000 output.pcm
```


