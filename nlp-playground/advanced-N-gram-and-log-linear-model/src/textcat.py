#!/usr/bin/env python3

"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.
"""

import argparse
import logging
import math
from pathlib import Path
import torch
from torch import Tensor
from jaxtyping import Float

from probs import Vocab, Wordtype, LanguageModel, read_trigrams

from typing import Counter, List

log = logging.getLogger(
    Path(__file__).stem
)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--models",
        dest="models",
        type=Path,
        help="path to the trained models.",
        nargs="*",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--priors",
        dest="prior_probability",
        type=float,
        help="the prior-probability for each model. "
        "If N-1 priors are provided for N models, the last is the complement. "
        "If no priors are provided, a uniform distribution is assumed.",
        nargs="*",
    )

    parser.add_argument(
        "-t",
        "--test_files",
        dest="test",
        type=Path,
        nargs="*",
        help="Path to files to classify",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)",
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
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        dest="logging_level",
        action="store_const",
        const=logging.WARNING,
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob: float = 0.0

    x: Wordtype
    y: Wordtype
    z: Wordtype  # type annotation for loop variables below
    for x, y, z in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file
        # probability to 0 and our cumulative log_prob to -infinity.  In
        # this case we can stop early, since the file probability will stay
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf:
            break

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


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

    n_models = len(args.models)
    priors: Float[Tensor, "models"]

    # If no prior is provided then we consider each model as having equal prior
    if args.prior_probability:
        # Allow user to specify N-1 priors, and we'll calculate the last one.
        if n_models - len(args.prior_probability) == 1:
            log.info("Found N-1 priors, calculating the complement for the Nth model.")
            complement = sum(args.prior_probability)
            if complement >= 1.0:
                raise ValueError("Sum of N-1 priors must be less than 1.")
            args.prior_probability.append(1.0 - complement)

        # At this point, the number of priors must match the number of models.
        if len(args.prior_probability) != n_models:
            raise ValueError(
                f"Number of priors ({len(args.prior_probability)}) must match the number of models ({n_models})."
            )

        priors = torch.tensor(args.prior_probability, dtype=torch.float)

    else:
        # No priors provided, so assume a uniform distribution.
        log.info("No priors provided, assuming a uniform distribution.")
        priors = torch.full((n_models,), 1.0)

    # Normalize priors to ensure they sum to 1 (in case user provides weights)
    priors /= priors.sum()
    log_priors = torch.log(priors)

    # NOTE: textcat differs from fileprob by doing what fileprob.py does n times for each given model
    # then picking the most probable one

    log.info("Testing each model...")
    # Store each model as list
    models: List[LanguageModel] = []
    mod: Path
    for mod in args.models:
        models.append(LanguageModel.load(mod, device=args.device))

        # Make sure that the Vocabularies used by each language model is the same
        if len(models) > 1:
            assert models[-1].vocab == models[-2].vocab, (
                "LanguageModels that are getting compared must have the same vocabulary"
            )

    # counter to know what percentage of the text files are classified under each model
    distribution: Counter = Counter()

    # iterate over all files and classify them by how likely
    # the text is being generated by a language model
    file: Path
    for file in args.test:
        # Used to store the score for each model.
        # Since we are doing a classification all we need is to Calculate the
        # numerator since the denominator is shared
        models_score: List[float] = []

        # iterate over each model to Calculate the cross-entropy for each
        m: LanguageModel
        for i, m in enumerate(models):
            # Calculate the cross-entropy for the file under all our models
            likelihood: float = file_log_prob(file, m)

            # Get the log prior
            log_model_prior: float = log_priors[i].item()

            models_score.append(likelihood + log_model_prior)

        best_index: int = models_score.index(max(models_score))

        # classify
        print(f"{file.stem}:\t{args.models[best_index].stem}")
        distribution[args.models[best_index]] += 1

    assert len(args.test) == sum(value for value in distribution.values())

    # print final summuries
    for key, values in distribution.items():
        print(f"There are {values} files classified a {key.stem}")


if __name__ == "__main__":
    main()
