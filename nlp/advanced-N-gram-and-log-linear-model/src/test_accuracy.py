#!/usr/bin/env python3
"""
Tests the accuracy of an output file
"""

import argparse
from collections import deque
import logging
from pathlib import Path
from typing import Counter, Deque, List, Dict

Summary_key = str

# Set logging level
log = logging.getLogger(
    Path(__file__).stem
)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        dest="path",
        type=Path,
        help="Path to the output file to be analyzed",
    )

    parser.set_defaults(last_n=2)
    parser.add_argument(
        "-d",
        "--disregard_n",
        dest="last_n",
        type=int,
        required=False,
        help="Last N lines to disregard in the line. Perhaps to neglect summuries printed in the output file",
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

    # ---- API To access dictionary
    correct_label: Summary_key = (
        "correct_label"  # Total number of correct classifications
    )
    total_label: Summary_key = "total_label"  # Total numbejjjjjjjjjjjr of classifications (both correct and incorrect)
    total_spam_label: Summary_key = "spam"  # Total number of spam classification
    total_gen_label: Summary_key = "gen"  # Total genuine classification
    spam_missclassification: Summary_key = (
        "spam_miss"  # number of incorrect files assigned as spam
    )
    gen_missclassification: Summary_key = (
        "gen_miss"  # number of incorrect files assigned as genuine
    )
    spam_documents_read: Summary_key = "spam_docs"
    gen_documents_read: Summary_key = "gen_docs"

    summuries: Dict[Summary_key, int] = Counter()

    disregard_last_n = args.last_n

    # File path is passed as an argument
    with open(args.path) as file:
        # Use a queue of size n lines and continue to pop from it. Last n lines wouldn't be read from
        queue: Deque[str] = deque()

        for line in file:
            queue.append(line)

            if len(queue) > disregard_last_n:
                # Pop and process the line
                classification_line: List[str] = queue.popleft().strip().split()

                summuries[total_label] += 1

                document: str = classification_line[0].lower()[:3]
                classification: str = classification_line[1].lower()[:3]

                if document == classification:
                    summuries[correct_label] += 1
                elif document == "gen":
                    summuries[gen_missclassification] += 1
                else:
                    summuries[spam_missclassification] += 1

                if classification_line == "gen":
                    summuries[total_gen_label] += 1
                else:
                    summuries[total_spam_label] += 1

                if document == "gen":
                    summuries[gen_documents_read] += 1
                else:
                    summuries[spam_documents_read] += 1

    assert (
        summuries[total_label] - summuries[correct_label]
        == summuries[spam_missclassification] + summuries[gen_missclassification]
    ), "Logic error in accuracy calculator"

    # Print summuries
    print(f"There are {summuries[total_label]} documents classified")
    print(f"{summuries[gen_documents_read]} of the document are genuine")
    print(f"{summuries[spam_documents_read]} of the documents are spam")
    print(
        f"There are {summuries[correct_label]} correct classification giving an accuracy of {summuries[correct_label] / summuries[total_label]:.2%}"
    )
    print(
        f"{summuries[spam_missclassification]} spam documents were missclassified as genuine"
    )
    print(
        f"{summuries[gen_missclassification]} genuine documents were missclassified as spam"
    )


if __name__ == "__main__":
    main()
