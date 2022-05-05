from pathlib import Path
import unittest

from random_segment import VocabControlledRandomSegmenter
from rich import print

BROWN_WORDLIST = Path("data/brown_wordlist.txt")


class TestControlledSegmentation(unittest.TestCase):
    def test_train_from_brown_wordlist(self):

        with open(BROWN_WORDLIST, "r", encoding="utf-8") as brown:
            count_word_tuples = (t.split() for t in brown)
            words = [word for count, word in count_word_tuples if word.strip()]

        rseg = VocabControlledRandomSegmenter(vocab_size=500, sep=" ")
        rseg.train(words=words)

        print("Here are some words and their segmentations:")

        for word in words[:5]:
            print(
                "Word: {}, Segmented: {}".format(word, rseg(word, return_as_list=True))
            )

        print("\nNow let's try a sentence:")
        sentence = "The quick brown fox jumped over the lazy dog, said Mary."
        segmented_sentence = rseg(sentence, return_as_list=False)
        reconstructed = segmented_sentence.replace(" ", "").replace(
            rseg.space_underscore, " "
        )
        print(f"Sentence: {sentence}")
        print(f"Segmented: {segmented_sentence}")
        print(f"Reconstructed: {reconstructed}")


if __name__ == "__main__":
    unittest.main(verbosity=1234)
