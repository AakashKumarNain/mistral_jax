from typing import List
from sentencepiece import SentencePieceProcessor


class MistralTokenizer:
    def __init__(self, model_path: str):
        self._model = SentencePieceProcessor(model_file=model_path)

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)
