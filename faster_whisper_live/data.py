from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from faster_whisper.transcribe import Word


class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    partial: bool

