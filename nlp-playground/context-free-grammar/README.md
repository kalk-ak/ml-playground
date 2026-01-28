# Context-Free Grammar Playground

This repository contains a set of tools for experimenting with context-free grammars (CFGs). It provides scripts to generate random sentences from a given grammar, parse text against a grammar, and format the output.

## Core Features

*   **Grammar-based Sentence Generation:** Create random sentences that adhere to the rules of your defined grammar.
*   **Parsing:** Check if a given sentence or text conforms to a specified grammar.
*   **Pretty Printing:** Format parsing results and grammar structures for better readability.
*   **Dynamic Parsing:** (Functionality to be explored via `dynaparse`)

## File Structure

*   `*.gr`, `*.gr_ec`: Grammar definition files. These files contain the rules of the context-free grammar.
*   `randsent.py`: A Python script to generate random sentences from a grammar.
*   `parse`: An executable or script to parse input text against a grammar.
*   `prettyprint`: An executable or script to format the output of the parser.
*   `dynaparse`: An executable or script for dynamic parsing.
*   `output.txt`, `prettier.txt`: Example output files.

## Usage

The primary tools in this repository are command-line based.

### Generating Random Sentences

Use the `randsent.py` script to generate a random sentence from a grammar file.

**Example:**
```bash
# Generate a random sentence using grammar.gr
python randsent.py grammar.gr
```

### Parsing Sentences

Use the `parse` script to check if an input sentence is valid according to a grammar.

**Example:**
```bash
# Parse a string from standard input
echo "the cat sat on the mat" | ./parse grammar.gr
```

### Pretty Printing

The `prettyprint` script can be used to format the output from the parser.

**Example:**
```bash
# Pipe the output of the parser to the pretty printer
echo "the cat sat on the mat" | ./parse grammar.gr | ./prettyprint
```

## Setup & Dependencies

The `.py` scripts require a Python interpreter. The other executables (`parse`, `prettyprint`, `dynaparse`) may need to be compiled or have specific execution permissions.

To make the scripts executable:
```bash
chmod +x parse prettyprint dynaparse
```
