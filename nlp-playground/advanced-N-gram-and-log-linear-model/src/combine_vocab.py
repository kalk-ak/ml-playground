#!/usr/bin/env python3
from pathlib import Path
import argparse
import logging

NAMESPACE = argparse.Namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "file1",
        type=Path,
        default=None,
        help="Path to the input CSV file (e.g., spam_Emails_data.csv)",
    )

    parser.add_argument(
        "file2",
        default=None,
        type=Path,
        help="Path to the input CSV file (e.g., spam_Emails_data.csv)",
    )

    parser.add_argument(
        "dest",
        default="output.txt",
        type=Path,
        help="Path to the input CSV file (e.g., spam_Emails_data.csv)",
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


def main() -> None:
    args: NAMESPACE = parse_args()

    assert args.file1 is not None and args.file2 is not None, (
        "No file pathes are provided"
    )
    with open(args.file1, "r") as file, open(args.dest, "w") as dest:
        for line in file:
            dest.write(line)

    with open(args.file2, "r") as file, open(args.dest, "a") as dest:
        for line in file:
            dest.write(line)

    logging.info(f"{args.file1} and {args.file2} written to destination {dest}")


if __name__ == "__main__":
    main()
