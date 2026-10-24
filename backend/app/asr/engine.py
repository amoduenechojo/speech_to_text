import numpy as np
from faster_whisper import WhisperModel

from .types import Word


class ASREngine:
    """Loads faster-whisper once and runs inference on in-memory audio windows."""

    def __init__(
        self,
        model_size: str,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        model: WhisperModel | None = None,
    ):
        self.language = language
        self.model = model or WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, initial_prompt: str = "") -> list[Word]:
        """Transcribe a float32 mono PCM window at 16kHz, returning word-level timings.

        `initial_prompt` is the already-committed text, passed as context so the
        model doesn't re-derive words it already produced.
        """
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt or None,
            vad_filter=False,
        )

        words: list[Word] = []
        for segment in segments:
            if not segment.words:
                continue
            for w in segment.words:
                words.append(Word(text=w.word, start=w.start, end=w.end))
        return words
