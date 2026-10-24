from dataclasses import dataclass


@dataclass
class Word:
    """A single transcribed word with its position in the source audio.

    `text` preserves the model's original spacing (usually a leading
    space) so that joining a list of words reproduces natural text.
    """

    text: str
    start: float
    end: float
