from typing import Optional, Iterable, Union, Tuple, Set, Sequence, List
from pathlib import Path
import random
import pickle

import attr
from rich.progress import track

import abc


class RandomSegmenter(abc.ABC):
    @abc.abstractmethod
    def segment_word(self, word: str, return_as_list: bool = False) -> str:
        raise NotImplementedError

    def __call__(self, word: str, return_as_list: bool = False) -> str:
        return self.segment_word(word, return_as_list=return_as_list)


@attr.s(auto_attribs=True)
class VocabControlledRandomSegmenter(RandomSegmenter):

    vocab_size: int = 0
    exclude_original_symbols: bool = False
    sep: str = " "
    trained: bool = False
    merges: list = attr.ib(default=[], repr=False)
    space_underscore = "\u2581"

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump(attr.asdict(self), f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VocabControlledRandomSegmenter":
        with open(path, "rb") as f:
            spec = pickle.load(f)

            return cls(**spec)

    def train(
        self,
        words: Optional[Union[Iterable[str], Tuple[str, int]]] = None,
        symbol_bigram_set: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Trains a new random segmentation model based on a word list."""
        assert (
            words or symbol_bigram_set
        ), "Must provide symbol_bigram_set or words to learn char_set from!"

        if not symbol_bigram_set:
            symbol_bigram_set = self.get_symbol_bigrams(words)

        merge_ops = []

        for ix in track(
            range(self.vocab_size),
            total=self.vocab_size,
            description=f"Randomly choosing {self.vocab_size} merges...",
        ):
            a, b = random.choice(symbol_bigram_set)
            merge_ops.append((f"{a} {b}", f"{a}{b}"))
            self.update_symbols(symbol_bigram_set, a, b)

        self.merges = merge_ops
        self.trained = True

    def update_symbols(
        self, symbol_bigram_set: List[Tuple[str, str]], a: str, b: str
    ) -> None:
        combined = f"{a}{b}"

        for ix, (first, second) in enumerate(symbol_bigram_set):
            if (first, second) == (a, b):
                symbol_bigram_set.pop(ix)
                ix -= 1

                continue
            elif first == b:
                new_tuple = (combined, second)
            elif second == a:
                new_tuple = (first, combined)
            else:
                new_tuple = (first, second)

            symbol_bigram_set[ix] = new_tuple

    @property
    def subword_vocabulary(self) -> List[str]:
        return [b for a, b in self.merges]

    def get_symbol_bigrams(self, words) -> List[Tuple[str, str]]:
        counts_provided = isinstance(words[0], tuple)
        word_iterable = words if counts_provided else ((w, 1) for w in words)

        def bigrams(word: str) -> Iterable[Tuple[str, str]]:
            return zip(word, word[1:])

        symbol_bigrams: Set[Tuple[str, str]] = set()

        for word, word_count in word_iterable:
            symbol_bigrams.update(bigrams(word))

        return sorted(symbol_bigrams)

    def segment_word(self, word: str, return_as_list: bool = False) -> str:
        assert (
            self.trained
        ), "Random segmentation with vocabulary control requires training!"

        return self._random_segment_controlled(word, return_as_list=return_as_list)

    def _random_segment_controlled(self, word: str, return_as_list: bool = False):
        _word = " ".join(c for c in word.replace(" ", self.space_underscore))

        for pattern, replacement in self.merges:
            _word = _word.replace(pattern, replacement)

        tokens = _word.split(" ")

        return (
            
            tokens

            if return_as_list
            else self.sep.join(tokens)
        )


@attr.s(auto_attribs=True)
class UncontrolledRandomSegmenter(RandomSegmenter):

    space_underscore = "\u2581"
    sep = "+"

    def segment_word(self, word) -> str:
        return self._random_segment_uncontrolled(word)

    def _should_split(self, p_split: float = 0.5) -> int:
        """Make a random decision about whether a split should occur"""

        return round(random.random() > p_split)

    def _random_segment_uncontrolled(self, word: str, sep: str = "+") -> str:
        """Segments a word with equal likelihood for all subword sequences"""

        return "".join(f"{sep}{c}" if self._should_split() else c for c in word).lstrip(
            sep
        )
