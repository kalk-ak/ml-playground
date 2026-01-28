#!/usr/bin/env python3
"""
Genreates random sentence from a given trigram model
"""

import argparse
import logging
import math
from pathlib import Path
import re
import torch
from jaxtyping import Float, Int

from probs import Vocab, Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(
    Path(__file__).stem
)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "-n",
        "--n_sentence",
        type=int,
        default=10,
        required=False,
        help="Number of sentences to generate (default 10).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        required=False,
        help="maximum length of tokens to generate for a given sentence. Rest of sentences is subistitued with '...'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        required=False,
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)",
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.WARNING)
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

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == "mps":
        if not torch.backends.mps.is_available():  # pyright: ignore[reportAttributeAccessIssue]
            if not torch.backends.mps.is_built():  # pyright: ignore[reportAttributeAccessIssue]
                logging.critical(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                logging.critical(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
            exit(1)
    torch.set_default_device(args.device)

    log.info(f"Generating {args.n_sentence} sentence ....")
    lm: LanguageModel = LanguageModel.load(args.model, device=args.device)

    generated_sentence: str = lm.sample(args.n_sentence, args.max_length, args.device)
    print(generated_sentence)


if __name__ == "__main__":
    main()
