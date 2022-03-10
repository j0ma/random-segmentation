from typing import Optional, Iterable, Any
import random

import attr
from tqdm import tqdm
import numpy as np


@attr.s(auto_attribs=True)
class RandomSegmenter:

    vocab_size: int = 0
    exclude_original_symbols: bool = False
    sep = "+"
    trained: bool = False
    merges: list = []
    space_underscore = "\u2581"

    def train(
        self,
        words: Optional[Iterable[str]] = None,
        symbol_bigram_set: Optional[set] = None,
    ) -> None:
        assert (
            words or symbol_bigram_set
        ), "Must provide symbol_bigram_set or words to learn char_set from!"

        if not symbol_bigram_set:
            symbol_bigram_set = self.get_symbol_bigram_set(words)

        merge_ops = []

        for ix in tqdm(range(self.vocab_size)):
            a, b = np.random.choice(symbol_bigram_set)
            merge_ops.append((f"{a} {b}", f"{a}{b}"))

        self.merges = merge_ops

    def segment_word(self, word) -> str:
        if not self.vocab_size:
            return self._random_segment_uncontrolled(word)
        else:
            assert (
                self.trained
            ), "Random segmentation with vocabulary control requires training!"

            return self._random_segment_controlled(word)

    def _random_segment_controlled(self, word: str, sep: str = "+"):
        _word = " ".join(c for c in word.replace(" ", self.space_underscore))
        print(_word)

        for pattern, replacement in self.merges:
            print(f"Replace {pattern} with {replacement} in {_word}")
            _word = _word.replace(pattern, replacement)

        print(_word)

        return "".join(f"{sep}{sw}" for sw in _word.split(" "))

    def _should_split(self, p_split: float = 0.5) -> int:
        """Make a random decision about whether a split should occur"""

        return round(random.random() > p_split)

    def _random_segment_uncontrolled(self, word: str, sep: str = "+") -> str:
        """Segments a word with equal likelihood for all subword sequences"""

        return "".join(f"{sep}{c}" if self._should_split() else c for c in word).lstrip(
            sep
        )
