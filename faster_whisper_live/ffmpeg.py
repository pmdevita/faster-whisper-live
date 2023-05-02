import asyncio
import subprocess
import threading
import traceback
import typing


class FFmpegDecoder:
    def __init__(self, file: typing.Union[typing.BinaryIO, str], ffmpeg="ffmpeg"):
        self.is_file = not isinstance(file, str)
        args = [ffmpeg, "-loglevel", "quiet",
                "-i", "pipe:0" if self.is_file else file,
                "-f", "s16le", "-ac", "1", "-ar", "16000",
                "-"]
        stdin = subprocess.PIPE if self.is_file else None
        self.p = subprocess.Popen(args, stdin=stdin, stdout=subprocess.PIPE)
        self.file = file
        self.should_terminate = False
        if self.is_file:
            print('starting thread')
            self.send_thread = threading.Thread(target=self.send)
            self.send_thread.start()

    def send(self):
        chunk = self.file.read(1024)
        while not self.should_terminate:
            print("chunk")
            self.p.stdin.flush()
            print("chunk 2")
            self.p.stdin.write(chunk)
            print("chunk 3")
            chunk = self.file.read(1024)
            print("chunk 4")
        self.p.terminate()

    def read(self, chunk_length: int):
        return self.p.stdout.read(chunk_length)

    def is_done(self):
        return self.p.poll() is not None

    def __del__(self):
        self.terminate()

    def terminate(self):
        self.should_terminate = True
        self.send_thread.join()


class AsyncFFmpegDecoder:
    def __init__(self, file: typing.Union[asyncio.StreamReader, str], ffmpeg="ffmpeg"):
        self.is_file = not isinstance(file, str)
        self.ffmpeg = ffmpeg
        self.file = file
        self.send_task = None
        self.p = None

    async def start(self):
        # "-loglevel", "quiet",
        args = [self.ffmpeg,
                "-i", "pipe:0" if self.is_file else file,
                "-f", "s16le", "-ac", "1", "-ar", "16000",
                "-"]
        stdin = asyncio.subprocess.PIPE if self.is_file else None
        self.p = await asyncio.create_subprocess_exec(*args, stdin=stdin, stdout=asyncio.subprocess.PIPE)
        if self.is_file:
            self.send_task = asyncio.Task(self.send())

    async def send(self):
        try:
            chunk = await self.file.read(1024)
            print("start send")
            while chunk:
                await self.p.stdin.drain()
                self.p.stdin.write(chunk)
                chunk = await self.file.read(1024)
        except asyncio.CancelledError:
            pass
        except:
            traceback.print_exc()

    async def read(self, chunk_length: int):
        print("reading from ffmpeg")
        return await self.p.stdout.read(chunk_length)

    def is_done(self):
        return self.p.returncode is not None

    def __del__(self):
        self.terminate()

    def terminate(self):
        self.p.terminate()
