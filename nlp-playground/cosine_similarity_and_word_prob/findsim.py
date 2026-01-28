#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer  # look at integerize.py for more info
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("\nERROR! You need to install Miniconda")
    raise


# using a logger
# Logging my code so that it is easier to debug
log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", type=Path, help="path to word embeddings file")
    parser.add_argument("word", type=str, help="word to look up")
    parser.add_argument(
        "-k", "--topk", type=int, help="topk closest to word", default=10
    )
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        dest="logging_level",
        action="store_const",
        const=logging.WARNING,
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error(f"Embeddings file {args.embeddings} not found")

    if (args.minus is None) ^ (
        args.plus is None
    ):  # Using the XOR operator because only one must be provided
        parser.error("Must include both `--plus` and `--minus` or neither")

    return args


class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words("bagpipe")
    """

    def __init__(self, word_list: List[str], embeddings: List[List[float]]) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""

        self._embdding_matrix = torch.tensor(
            embeddings, dtype=torch.float64
        )  # Placeholder
        logging.log(
            2, "created embedding matrix of size %s", self._embdding_matrix.size()
        )

        self.integerizer = Integerizer(word_list)  # Placeholder
        logging.log(
            2, "Initialized an Integerizer that has %d words", len(self.integerizer)
        )

        self._embdding_matrix = nn.functional.normalize(
            self._embdding_matrix, p=2, dim=1
        )
        logging.debug(
            "normalizing the embedding matrix so that cosine similarity is just dot product"
        )

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        with open(file) as f:
            first_line = next(f)  # Peel off the special first line.
            len_vocabulary, embedding_dim = map(int, first_line.split())

            # Populate the dynamic data structures. Instead of the fixed size tensor
            word_list: List[str] = []
            embedding_list: List[List[float]] = []

            logging.debug("Reading lines")
            for line in f:  # All of the other lines are regular.
                line_list: List[str] = (
                    line.strip().split()
                )  # Create a list to store the tokenized line

                word: str = line_list[
                    0
                ]  # the word is stored as the first element in the row

                embedding: List[float] = [
                    float(token) for token in line_list[1:]
                ]  # the embedding is the rest of the row

                word_list.append(word)
                embedding_list.append(embedding)

        logging.debug("Creating lexicon for known words")
        lexicon = Lexicon(
            word_list, embedding_list
        )  # Maybe put args here. Maybe follow Builder pattern.
        return lexicon

    def find_similar_words(
        self,
        word: str,
        k: int = 1,
        *,
        plus: Optional[str] = None,
        minus: Optional[str] = None,
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""

        if (minus is None) ^ (plus is None):
            raise TypeError("Must include both of `plus` and `minus` or neither.")

        # Check if the word is in the Lexicon (is known)
        word_id: Optional[int] = self.integerizer.index(word)
        assert word_id is not None, f"The word '{word}' is not in the lexicon."

        # Index the word from the embedding matrix
        row: torch.Tensor = self._embdding_matrix[word_id]

        # INFO: Added functionality for adding and subtracting words
        if (minus is not None) and (plus is not None):
            minus_id: Optional[int] = self.integerizer.index(minus)
            plus_id: Optional[int] = self.integerizer.index(plus)

            assert minus_id is not None or plus_id is not None, (
                "Plus or Minus word not in lexicon."
            )

            # get the rows for plus and minus
            plus_row: torch.Tensor = self._embdding_matrix[plus_id]
            minus_row: torch.Tensor = self._embdding_matrix[minus_id]

            row += plus_row - minus_row

        # Calculate the dot product for the all the embedding vectors representing words
        similarities: torch.Tensor = self._embdding_matrix @ row

        # Make sure that we don't pick the query vector as our final answer
        similarities[word_id] = -float("inf")

        _, top_index = torch.topk(
            similarities,
            k,
        )

        # Get the top k similar_words
        top_smilar: List[str] = [self.integerizer[int(i)] for i in top_index]

        return top_smilar


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus, k=args.topk
    )
    print(" ".join(similar_words))  # print all words on one line, separated by spaces


if __name__ == "__main__":
    main()
