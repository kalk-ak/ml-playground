#!/usr/bin/env python3

import argparse
import os
import random
import subprocess
import sys
from pyperclip import copy

# my additions:
from typing import TypedDict, List, Tuple
from collections import defaultdict

# <<< Added DEBUG flag
DEBUG = True

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
        self.expansions_done = 0

        def expand_nonterminal(nonterminal: str) -> Tuple[str, str]:
            """
            Recursively expands a nonterminal.
            Returns a tuple of (plain_string, tree_string).
            """
            if self.expansions_done >= max_expansions:
                return ("...", "...")

            is_terminal = nonterminal not in self.rules
            if is_terminal:
                return (nonterminal, nonterminal)

            self.expansions_done += 1
            rhs = self.randomly_choose_nonterminal_expansion(nonterminal)

            plain_parts = []
            tree_parts = ["(" + nonterminal]

            for symbol in rhs.split():
                plain, tree = expand_nonterminal(symbol)
                plain_parts.append(plain)
                tree_parts.append(tree)

            tree_parts.append(")")

            plain_string = " ".join(plain_parts)
            tree_string = " ".join(tree_parts)

            return (plain_string, tree_string)

        plain_final, tree_final = expand_nonterminal(start_symbol)
        copy(plain_final)

        if DEBUG:
            # When debugging, we print both, so we need to handle the tree string
            # to be pretty-printed if the user asked for it.
            if derivation_tree:
                return f"{plain_final}\n{tree_final}"
            else:
                return f"{plain_final}\n{tree_final}"
        else:
            return tree_final if derivation_tree else plain_final


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
        if args.tree and not DEBUG:
            prettyprint_path = os.path.join(os.getcwd(), "prettyprint")
            subprocess.run(["perl", prettyprint_path], input=sentence, text=True)
        elif DEBUG:
            parts = sentence.split("\n")
            plain_sentence = parts[0]
            tree_sentence = parts[1]
            print(plain_sentence)
            prettyprint_path = os.path.join(os.getcwd(), "prettyprint")
            subprocess.run(["perl", prettyprint_path], input=tree_sentence, text=True)
        else:
            print(sentence)


if __name__ == "__main__":
    main()
