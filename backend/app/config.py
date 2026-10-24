from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # audio
    sample_rate: int = 16000

    # ASR
    # large-v3-turbo is the target for production (see design doc), but it's a
    # ~1.6GB download; default to "small" for local dev as a middle ground —
    # noticeably better diction than "base" without the huge fetch. Override
    # with SPEAKLIFE_MODEL_SIZE.
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"

    # streaming / stabilizer
    max_buffer_seconds: float = 15.0
    step_seconds: float = 1.0

    # VAD
    vad_threshold: float = 0.5
    vad_frame_samples: int = 512  # silero-vad requires 512 samples per frame at 16kHz

    # logging
    debug: bool = False  # per-chunk/per-transcript logging; noisy, keep off in production

    model_config = {"env_prefix": "SPEAKLIFE_"}


settings = Settings()