import numpy as np

from app.asr.stream_processor import StreamProcessor
from app.asr.types import Word


class FakeEngine:

    def __init__(self, hypotheses):
        self._hypotheses = hypotheses
        self.calls = 0

    def transcribe(self, audio, initial_prompt=""):
        words = self._hypotheses[min(self.calls, len(self._hypotheses) - 1)]
        self.calls += 1
        return words


def make_words(spec):
    words = []
    t = 0.0
    for w in spec:
        words.append(Word(text=f" {w}", start=t, end=t + 0.3))
        t += 0.3
    return words


def make_processor(engine, **overrides):
    kwargs = dict(sample_rate=16000, max_buffer_seconds=10, min_step_seconds=0.01)
    kwargs.update(overrides)
    return StreamProcessor(engine=engine, **kwargs)


def test_commits_agreed_prefix_and_keeps_tail_pending():
    hypotheses = [
        make_words(["hello", "wor"]),
        make_words(["hello", "world", "how"]),
    ]
    processor = make_processor(FakeEngine(hypotheses))

    processor.add_audio(np.zeros(1600, dtype=np.float32))
    first = processor.process()
    assert first["committed"] == ""

    processor.add_audio(np.zeros(1600, dtype=np.float32))
    second = processor.process()
    assert second["committed"].strip() == "hello"
    assert "world" in second["pending"]
    assert "how" in second["pending"]


def test_never_re_emits_committed_text():
    hypotheses = [
        make_words(["hello", "there"]),
        make_words(["hello", "there", "friend"]),
        make_words(["friend", "how", "are"]),
    ]
    processor = make_processor(FakeEngine(hypotheses))

    seen_committed = []
    for _ in range(3):
        processor.add_audio(np.zeros(1600, dtype=np.float32))
        result = processor.process()
        seen_committed.append(result["committed"])

    for earlier, later in zip(seen_committed, seen_committed[1:]):
        assert later.startswith(earlier)


def test_disagreement_keeps_everything_pending():
    hypotheses = [make_words(["cat"])]
    processor = make_processor(FakeEngine(hypotheses))

    processor.add_audio(np.zeros(1600, dtype=np.float32))
    result = processor.process()

    assert result["committed"] == ""
    assert result["pending"] == "cat"


def test_buffer_trims_to_max_length():
    processor = make_processor(FakeEngine([make_words(["a"])]), max_buffer_seconds=1)
    processor.add_audio(np.zeros(16000 * 3, dtype=np.float32))
    assert len(processor.buffer) <= 16000


def test_not_ready_until_min_step_reached():
    processor = make_processor(FakeEngine([make_words(["a"])]), min_step_seconds=1.0)
    processor.add_audio(np.zeros(100, dtype=np.float32))
    assert processor.ready() is False
