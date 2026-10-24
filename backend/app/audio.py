import numpy as np


def normalize_gain(chunk: np.ndarray, target_peak: float = 0.5, max_gain: float = 6.0) -> np.ndarray:
    """Boost quiet audio toward `target_peak` before it hits VAD/ASR.

    Speech recorded farther from the mic arrives at much lower amplitude, which
    can drop it below the VAD's speech-probability threshold entirely and gives
    Whisper a weaker signal to transcribe. `max_gain` caps how far near-silent
    audio (background noise, mic hiss) gets boosted, so it doesn't get amplified
    into false VAD triggers.
    """
    peak = float(np.abs(chunk).max()) if len(chunk) else 0.0
    if peak == 0.0:
        return chunk
    gain = min(target_peak / peak, max_gain)
    if gain <= 1.0:
        return chunk
    return chunk * gain
