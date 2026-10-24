from typing import Any

import numpy as np
import torch


class SileroVAD:
    """Wraps Silero VAD to gate audio chunks as speech/silence before they hit the ASR buffer.

    `frame_samples` must match what the underlying Silero model was trained on
    (512 samples per frame at 16kHz); arbitrary-length chunks are split into
    frames of that size before scoring.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        model: Any = None,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.model = model or self._load_model()

    @staticmethod
    def _load_model():
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        return model

    def speech_prob(self, frame: np.ndarray) -> float:
        tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate)
        return float(prob.item() if hasattr(prob, "item") else prob)

    def contains_speech(self, chunk: np.ndarray) -> bool:
        """True if any frame_samples-sized frame in `chunk` scores above threshold."""
        found = False
        for start in range(0, len(chunk) - self.frame_samples + 1, self.frame_samples):
            frame = chunk[start : start + self.frame_samples]
            if self.speech_prob(frame) >= self.threshold:
                found = True
        return found

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
