from pathlib import Path
import unittest

from random_segment import VocabControlledRandomSegmenter
from rich.progress import track
from rich import print

BROWN_WORDLIST = Path("data/brown_wordlist.txt")
ESTONIAN = Path("data/et_mono")
ESTONIAN_TAIL = Path("data/et_mono_tail")
ESTONIAN_TAIL_100k = Path("data/et_mono_tail_100k")
RSEG_SAVE_PATH = Path("/tmp/rseg_test")


class TestControlledSegmentation(unittest.TestCase):
    def test_train_from_brown_wordlist(self):

        with open(BROWN_WORDLIST, "r", encoding="utf-8") as brown:
            count_word_tuples = (t.split() for t in brown)
            words = [word for count, word in count_word_tuples if word.strip()]

        rseg = VocabControlledRandomSegmenter(vocab_size=500, sep=" ")

        print(f"\nTraining {rseg}...")
        rseg.train(words=words)
        print("Done! Here are some merges")
        print(rseg.merges[:10])

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
        print(f"Equal? {sentence == reconstructed}")

    def test_train_from_estonian_sentences(self):

        with open(ESTONIAN, "r", encoding="utf-8") as et_mono:
            sentences = [s.strip() for s in et_mono if s.strip()]

        rseg = VocabControlledRandomSegmenter(vocab_size=1000, sep=" ")
        print(f"\nTraining {rseg}...")
        rseg.train(words=sentences)
        print("Done! Here are some merges")
        print(rseg.merges[:10])

        print("Dump to disk:")
        rseg.save(path=RSEG_SAVE_PATH)
        del rseg
        print(f"No more rseg? {'rseg' not in globals() and 'rseg' not in locals()}")

        rseg = VocabControlledRandomSegmenter.load(RSEG_SAVE_PATH)
        print(f"Loaded: {rseg}")

        print("\nNow let's try a sentence:")
        sentence = "The quick brown fox jumped over the lazy dog, said Mary."
        segmented_sentence = rseg(sentence, return_as_list=False)
        reconstructed = segmented_sentence.replace(" ", "").replace(
            rseg.space_underscore, " "
        )
        print(f"Sentence: {sentence}")
        print(f"Segmented: {segmented_sentence}")
        print(f"Reconstructed: {reconstructed}")
        print(f"Equal? {sentence == reconstructed}")

        subwords = set()
        with open(ESTONIAN_TAIL, "r", encoding="utf-8") as et_mono_tail:
            for sent in track(
                et_mono_tail,
                total=50000,
                description="Applying to 50K Estoonian sentences",
            ):
                sent = sent.strip()
                segments = rseg(sent, return_as_list=True)
                subwords.update(segments)

        print(f"Subword vocabulary size: {len(subwords)}")

        subwords = set()
        with open(ESTONIAN_TAIL_100k, "r", encoding="utf-8") as et_mono_tail_100k:
            for sent in track(
                et_mono_tail_100k,
                total=100000,
                description="Applying to 100K Estoonian sentences",
            ):
                sent = sent.strip()
                segments = rseg(sent, return_as_list=True)
                subwords.update(segments)

        print(f"Subword vocabulary size: {len(subwords)}")


if __name__ == "__main__":
    unittest.main(verbosity=1234)
