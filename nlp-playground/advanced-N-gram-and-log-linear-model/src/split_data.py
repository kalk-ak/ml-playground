#!/usr/bin/env python3

"""
Splits a kaggle dataset into seperate folders separated by gen and spam
"""

from pathlib import Path
import argparse
import logging
import csv
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input CSV file (e.g., spam_Emails_data.csv)",
    )

    parser.add_argument(
        "-c",
        "--no-combine",
        dest="combine_file",
        action="store_false",
        default=True,
        help="Do NOT combine emails into two large 'combined' files. (Default: combining is enabled).",
    )

    parser.add_argument(
        "-s",
        "--no-split",
        dest="split_across_directory",
        action="store_false",
        default=True,
        help="Do NOT split emails into individual files. (Default: splitting is enabled).",
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


def main():
    # Increase the CSV field size limit to handle very large email bodies.
    # The default is 131072 (128KB). We'll increase it to ~100MB.
    csv.field_size_limit(100 * 1024 * 1024)

    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    log = logging.getLogger(__name__)

    log.info(f"Input file: {args.input_file}")
    log.info(f"Combine into single files: {args.combine_file}")
    log.info(f"Split into individual files: {args.split_across_directory}")

    if not args.combine_file and not args.split_across_directory:
        log.warning(
            "Both --no-combine and --no-split were specified. No output will be generated."
        )
        return

    # Define and create output directories
    output_base_dir = Path("data")
    gen_dir = output_base_dir / "kaggle_gen"
    spam_dir = output_base_dir / "kaggle_spam"
    gen_dir.mkdir(parents=True, exist_ok=True)
    spam_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Ensured directories exist: {gen_dir}, {spam_dir}")

    # Prepare file handles for combined output, if requested
    gen_combined_file = None
    spam_combined_file = None

    try:
        if args.combine_file:
            gen_combined_file = (gen_dir / "combined").open("w", encoding="utf-8")
            spam_combined_file = (spam_dir / "combined").open("w", encoding="utf-8")
            log.info("Opened combined files for writing.")

        with args.input_file.open("r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            header = next(reader, None)  # Read header row
            if not header:
                log.warning("Input CSV file is empty.")
                return
            log.info(f"CSV Header: {header}")

            for line_number, row in enumerate(reader, 1):
                if len(row) < 2:
                    log.warning(f"Row {line_number} has < 2 columns, skipping: {row}")
                    continue

                label = row[0].strip().lower()
                text = row[1].strip()

                # Determine where to write based on the label
                if label == "ham":
                    if args.combine_file:
                        gen_combined_file.write(text + "\n")  # pyright: ignore[reportOptionalMemberAccess]
                    if args.split_across_directory:
                        with (gen_dir / f"gen.{line_number}.txt").open(
                            "w", encoding="utf-8"
                        ) as outfile:
                            outfile.write(text)
                elif label == "spam":
                    if args.combine_file:
                        spam_combined_file.write(text + "\n")  # pyright: ignore[reportOptionalMemberAccess]
                    if args.split_across_directory:
                        with (spam_dir / f"spam.{line_number}.txt").open(
                            "w", encoding="utf-8"
                        ) as outfile:
                            outfile.write(text)
                else:
                    log.warning(
                        f"Unknown label '{label}' in row {line_number}, skipping."
                    )

    except FileNotFoundError:
        log.error(f"Input file not found: {args.input_file}")
        return
    except Exception as e:
        log.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Ensure combined files are closed safely
        if gen_combined_file:
            gen_combined_file.close()
        if spam_combined_file:
            spam_combined_file.close()
        log.info("Processing finished. All files closed.")


if __name__ == "__main__":
    main()
