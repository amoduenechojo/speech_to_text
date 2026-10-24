import numpy as np
import torch

from app.asr.vad import SileroVAD


class FakeModel:
    """Returns canned speech probabilities in call order, one per frame."""

    def __init__(self, values):
        self._values = values
        self.calls = 0

    def __call__(self, tensor, sample_rate):
        value = self._values[self.calls]
        self.calls += 1
        return torch.tensor(value)

    def reset_states(self):
        self.calls = 0


def test_silence_is_not_speech():
    vad = SileroVAD(threshold=0.5, frame_samples=512, model=FakeModel([0.1]))
    chunk = np.zeros(512, dtype=np.float32)
    assert vad.contains_speech(chunk) is False


def test_speech_above_threshold_is_detected():
    vad = SileroVAD(threshold=0.5, frame_samples=512, model=FakeModel([0.9]))
    chunk = np.zeros(512, dtype=np.float32)
    assert vad.contains_speech(chunk) is True


def test_chunk_is_split_into_fixed_size_frames():
    model = FakeModel([0.1, 0.1, 0.9])
    vad = SileroVAD(threshold=0.5, frame_samples=512, model=model)
    chunk = np.zeros(512 * 3, dtype=np.float32)

    assert vad.contains_speech(chunk) is True
    assert model.calls == 3


def test_reset_delegates_to_underlying_model():
    model = FakeModel([0.9])
    vad = SileroVAD(threshold=0.5, frame_samples=512, model=model)
    vad.contains_speech(np.zeros(512, dtype=np.float32))
    vad.reset()
    assert model.calls == 0
