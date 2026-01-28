#!/usr/bin/env python3

import argparse
import os
import random
import subprocess
import sys

# my additions:
from typing import TypedDict, List
from collections import defaultdict

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Generate random sentences from a PCFG"
    )
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str,
        required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file

        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = None
        self._load_rules_from_file(grammar_file)

    def randomly_choose_nonterminal_expansion(self, nonterminal) -> str:
        assert nonterminal in self.rules
        options = self.rules[nonterminal]

        return random.choices(
            population=[option["rhs"] for option in options],
            weights=[option["relative_odds"] for option in options],
            k=1,
        )[0]

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules.

        This version includes robust validation for:
        - Malformed lines (incorrect number of tabs)
        - Negative weights for rules
        - Invalid characters (whitespace, parentheses) in LHS/RHS symbols

        Args:
            grammar_file (str): Path to the raw grammar file
        """

        # initialize a rules dictionary
        class RuleValue(TypedDict):
            relative_odds: float
            rhs: str  # sequence of zero or more terminal and nonterminal symbols, space-separated

        rules: defaultdict[str, List[RuleValue]] = defaultdict(list)
        #                  ^^^ LHS nonterminal symbol

        with open(grammar_file, "r") as f:
            for line_num, line in enumerate(f.readlines(), 1):
                # first, deal with comments by looping over all lines and removing everything past "#"s
                comment_removed_line = ""
                for char in line.strip():
                    if char == "#":
                        break
                    comment_removed_line += char

                comment_removed_line = comment_removed_line.strip()

                # next, remove empty lines (after removing comments and stripping whitespace)
                if not comment_removed_line:
                    continue

                # VALIDATION: Check for the correct number of tabs. Replaces original assert.
                if comment_removed_line.count("\t") != 2:
                    print(
                        f"[ERROR] Line {line_num}: Malformed line found without exactly 2 tabs. Possibly because of use of '#' in grammar rule, which is illegal.",
                        file=sys.stderr,
                    )
                    continue

                # split by tabs to get into the 3 key parts:
                odds_string, lhs, rhs = comment_removed_line.split("\t")

                # check for negative weights.
                try:
                    if float(odds_string) < 0:
                        print(
                            f"[ERROR] Line {line_num}: Negative probability/weight '{odds_string}' found for rule '{lhs} -> {rhs}'. Skipping rule.",
                            file=sys.stderr,
                        )
                        continue
                except ValueError:
                    print(
                        f"[ERROR] Line {line_num}: Could not parse weight '{odds_string}' as a float. Skipping rule.",
                        file=sys.stderr,
                    )
                    continue

                # check that the LHS doesn't have whitespace or parentheses.
                if " " in lhs or "(" in lhs or ")" in lhs:
                    print(
                        f"[ERROR] Line {line_num}: Whitespace or parentheses found in LHS symbol '{lhs}'. Skipping rule.",
                        file=sys.stderr,
                    )
                    continue

                # If all checks pass, add the rule
                rules[lhs].append({"relative_odds": float(odds_string), "rhs": rhs})

        self.rules = dict(rules)  # convert from defaultdict to normal dict
        return self.rules

    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent
                the tree (using bracket notation) that records how the sentence
                was derived

            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        nonterminal_to_expand = start_symbol

        self.expansions_done = (
            0  # 'global' variable so we don't have to deal with it in recursion
        )

        def expand_nonterminal(nonterminal: str) -> str:
            if self.expansions_done >= max_expansions:
                return "..."

            nonnonterminal = (
                nonterminal not in self.rules
            )  # true: it's actually a terminal, not a nonterminal

            if not not not not nonnonterminal:  # if it's a terminal
                return nonterminal  # simply return the original string, regardless of derivation_tree

            # from here on out, the variable name nonterminal is accurate
            expanded_string = "(" + nonterminal if derivation_tree else ""
            rhs = self.randomly_choose_nonterminal_expansion(nonterminal)

            self.expansions_done += 1  # the below line is where we actually expand our nonterminal, so this is a fitting place to increment
            for potential_nonterminal in rhs.split():
                expanded = expand_nonterminal(potential_nonterminal)
                expanded_string += " " + expanded

            return (
                expanded_string + (")" if derivation_tree else "")
            ).strip()  # .strip() removes original space if not derivation_tree

        return expand_nonterminal(nonterminal_to_expand)


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol,
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), "prettyprint")
            subprocess.run(["perl", prettyprint_path], input=sentence, text=True)
        else:
            print(sentence)


if __name__ == "__main__":
    main()
