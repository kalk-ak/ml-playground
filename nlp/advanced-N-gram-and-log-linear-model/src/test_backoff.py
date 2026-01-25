#!/usr/bin/env python3


import logging

import argparse
import math
import random
from typing import List
from pathlib import Path

from probs import (
    BOS,
    LanguageModel,
    Trigram,
    Vocab,
    Wordtype,  #
    read_vocab,
    BackoffAddLambdaLanguageModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        dest="vocab_path",
        type=Path,
        help="Path to the vocabulary",
    )

    parser.add_argument(
        dest="model_path",
        type=Path,
        help="Path to the trained_model",
    )

    parser.set_defaults(n_checks=100)  # Default set to 100 different vocab checks
    parser.add_argument(
        "-n",
        "--n_checks",
        dest="n_checks",
        type=int,
        required=False,
        help="Specify how many random checks to make",
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
        required=False,
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        dest="logging_level",
        action="store_const",
        const=logging.WARNING,
        required=False,
    )

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Load the trained model from the specified path.
    # .load is a class method that returns a new, trained model instance.
    logging.info(f"Loading trained model from {args.model_path}")
    lm: LanguageModel = BackoffAddLambdaLanguageModel.load(args.model_path)

    assert isinstance(lm, BackoffAddLambdaLanguageModel), "Incorrect model type loaded"
    # The vocabulary for creating contexts can include BOS.
    # The second word of a trigram context (y) cannot be EOS.
    context_pool = list(lm.vocab) + [BOS]
    context_pool_no_eos = [w for w in context_pool if w != "EOS"]

    logging.info(f"Performing {args.n_checks} random probability checks...")

    for i in range(args.n_checks):
        # Reset the probability sum for each new context check
        total_trigram_prob: float = 0.0

        # Pick a random bigram context (x, y). Allow BOS in the context.
        x = random.choice(context_pool)
        y = random.choice(context_pool_no_eos)
        dummy_bigram: List[Wordtype] = [x, y]

        # Sum probabilities over all possible next words in the model's vocabulary
        for word in lm.vocab:
            trigram: Trigram = (dummy_bigram[0], dummy_bigram[1], word)
            total_trigram_prob += lm.prob(*trigram)

        # Assert that the sum is close to 1.0 for each context
        try:
            assert math.isclose(total_trigram_prob, 1.0)
        except AssertionError:
            logging.error(
                f"FAILURE on context ({x}, {y}): Probabilities sum to {total_trigram_prob}, not 1.0!"
            )
            # Exit on first failure to prevent flood of errors
            exit(1)

    logging.info(
        f"SUCCESS: All {args.n_checks} checks passed. Probability distributions sum to 1."
    )


if __name__ == "__main__":
    main()
