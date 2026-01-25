# Word Similarity with Embeddings

This project provides a Python script to find the most similar words to a given word using pre-trained word embeddings. It can also perform word analogy tasks, such as "king - man + woman = queen".

## Features

-   Find the top-k most similar words to a given word.
-   Perform word analogy tasks using vector arithmetic on word embeddings.
-   Efficiently handles large embedding files.

## Requirements

-   Python 3.6+
-   PyTorch

You can install PyTorch by following the instructions on the [official website](https://pytorch.org/get-started/locally/).

## Usage

The main script is `findsim.py`. You can run it from the command line.

### Basic Usage

To find the top 10 most similar words to "hello":

```bash
./findsim.py lexicon/words-50.txt hello
```

### Options

-   `embeddings`: (Required) Path to the word embeddings file. The file should be in a text format where the first line is `<vocabulary_size> <embedding_dimension>` and each subsequent line is a word followed by its embedding vector.
-   `word`: (Required) The word to find similar words for.
-   `-k` or `--topk`: The number of similar words to return (default: 10).
-   `--plus` and `--minus`: For word analogies. For example, to calculate "king - man + woman":

    ```bash
    ./findsim.py lexicon/words-50.txt king --minus man --plus woman
    ```

-   `-v` or `--verbose`: Enable verbose logging.
-   `-q` or `--quiet`: Suppress all but essential logging.

## Code Structure

-   **`findsim.py`**: The main script for finding similar words. It loads word embeddings, parses command-line arguments, and uses the `Lexicon` class to perform the similarity search.
-   **`integerize.py`**: A utility class `Integerizer` that maps objects (like words) to unique integers and back. This is used by the `Lexicon` class to efficiently manage the vocabulary.
-   **`lexicon/`**: A directory containing sample word embedding files.

## How it Works

1.  **Loading Embeddings**: The `Lexicon` class reads the word embeddings from the given file. It uses the `Integerizer` class to create a mapping from words to integers, which are used as indices for the embedding matrix.
2.  **Normalization**: The embedding matrix is L2-normalized. This means that the cosine similarity between two word vectors can be calculated by a simple dot product.
3.  **Similarity Search**: To find similar words, the script calculates the dot product of the query word's vector with all other word vectors in the vocabulary.
4.  **Top-k**: The `torch.topk` function is used to find the `k` words with the highest similarity scores.
5.  **Word Analogies**: For analogies like `a - b + c`, the script first computes the resulting vector by adding and subtracting the corresponding word vectors and then finds the words most similar to this new vector.
