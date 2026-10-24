import numpy as np

from app.audio import normalize_gain


def test_boosts_quiet_audio_toward_target_peak():
    chunk = np.array([0.05, -0.05, 0.02], dtype=np.float32)
    result = normalize_gain(chunk, target_peak=0.5, max_gain=6.0)
    assert np.abs(result).max() == np.float32(0.05 * 6.0)


def test_leaves_already_loud_audio_untouched():
    chunk = np.array([0.6, -0.4, 0.1], dtype=np.float32)
    result = normalize_gain(chunk, target_peak=0.5, max_gain=6.0)
    np.testing.assert_array_equal(result, chunk)


def test_caps_gain_so_near_silent_noise_is_not_amplified_unbounded():
    chunk = np.array([0.001, -0.0005], dtype=np.float32)
    result = normalize_gain(chunk, target_peak=0.5, max_gain=6.0)
    assert np.abs(result).max() == np.float32(0.001 * 6.0)


def test_silence_stays_silent():
    chunk = np.zeros(10, dtype=np.float32)
    result = normalize_gain(chunk)
    np.testing.assert_array_equal(result, chunk)


def test_empty_chunk_does_not_crash():
    chunk = np.zeros(0, dtype=np.float32)
    result = normalize_gain(chunk)
    assert len(result) == 0
