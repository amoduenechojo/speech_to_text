import numpy as np

from .types import Word


class StreamProcessor:
    """Buffers audio and turns repeated whole-window transcriptions into stable text.

    Whisper has no streaming mode, so every `process()` call re-transcribes the
    entire rolling buffer. To avoid visibly flickering captions, we only "commit"
    the word-level prefix that agrees across two consecutive transcriptions
    (LocalAgreement-2). Committed audio is then trimmed from the buffer so future
    inference only has to resolve the still-uncertain tail, and the committed
    text is fed back in as `initial_prompt` context.
    """

    def __init__(
        self,
        engine,
        sample_rate: int = 16000,
        max_buffer_seconds: float = 15.0,
        min_step_seconds: float = 1.0,
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.max_buffer_samples = int(max_buffer_seconds * sample_rate)
        self.min_step_samples = int(min_step_seconds * sample_rate)

        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_last_run = 0
        self.committed_text = ""
        self._last_hypothesis: list[Word] = []

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.samples_since_last_run = 0
        self.committed_text = ""
        self._last_hypothesis = []

    def add_audio(self, chunk: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, chunk])
        self.samples_since_last_run += len(chunk)
        if len(self.buffer) > self.max_buffer_samples:
            overflow = len(self.buffer) - self.max_buffer_samples
            self.buffer = self.buffer[overflow:]

    def ready(self) -> bool:
        return self.samples_since_last_run >= self.min_step_samples and len(self.buffer) > 0

    def process(self) -> dict:
        self.samples_since_last_run = 0

        hypothesis = self.engine.transcribe(self.buffer, initial_prompt=self.committed_text)
        agreed = self._agreement_prefix(self._last_hypothesis, hypothesis)
        self._last_hypothesis = hypothesis

        pending_words = hypothesis[len(agreed) :]

        if agreed:
            newly_committed = "".join(w.text for w in agreed)
            self.committed_text = (self.committed_text + newly_committed) if self.committed_text else newly_committed.lstrip()

            cut_sample = int(agreed[-1].end * self.sample_rate)
            cut_sample = min(cut_sample, len(self.buffer))
            self.buffer = self.buffer[cut_sample:]
            self._last_hypothesis = []

        return {
            "committed": self.committed_text,
            "pending": "".join(w.text for w in pending_words).strip(),
        }

    @staticmethod
    def _agreement_prefix(previous: list[Word], current: list[Word]) -> list[Word]:
        agreed = []
        for prev_word, cur_word in zip(previous, current):
            if prev_word.text == cur_word.text:
                agreed.append(cur_word)
            else:
                break
        return agreed
