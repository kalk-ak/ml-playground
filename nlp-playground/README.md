# NLP Playground: From Classic Models to Modern LLMs

Welcome to the NLP Playground! This repository is a curated collection of natural language processing projects, spanning a wide range of techniques from foundational concepts to contemporary language model applications. Whether you're interested in the mathematical underpinnings of statistical models or the practical implementation of dialogue agents, you'll find something to explore here.

This playground is designed for both learning and experimentation. Each project is self-contained with its own set of tools, data, and documentation, allowing you to dive deep into the areas that interest you most.

## 🚀 Project Highlights

This repository is divided into several distinct projects:

*   **LLM (Argubots):** A framework for creating and evaluating dialogue agents that can engage in structured debates. This project uses data from Kialo.com to power its "argubots."
*   **Advanced N-gram and Log-Linear Models:** A toolkit for building and evaluating statistical and neural language models. It includes implementations of classic smoothing techniques as well as modern log-linear models.
*   **Context-Free Grammar (CFG):** A set of tools for generating random sentences from a formal grammar and parsing text to see if it conforms to a grammar's rules.
*   **Cosine Similarity and Word Probability:** A project that explores word embeddings to find similar words and solve analogies (e.g., "king - man + woman = queen").

---

## 🤖 LLM (Argubots)

This project is a platform for developing and evaluating "argubots"—dialogue agents designed to engage in debates.

### Key Features:

*   **Variety of Agents:** Includes a range of argubots with different strategies.
*   **Kialo Integration:** Utilizes data from Kialo.com for structured debate arguments.
*   **Evaluation Framework:** A built-in system for assessing the performance of the argubots.
*   **Extensible:** Designed to be easily expanded with new agents and evaluation methods.

For more details, see the [`LLM/README.md`](LLM/README.md).

---

## 📊 Advanced N-gram and Log-Linear Models

A comprehensive toolkit for training, evaluating, and sampling from N-gram language models.

### Key Features:

*   **Statistical Models:** Implements Uniform, Add-Lambda, and Backoff Add-Lambda smoothing.
*   **Neural Models:** Includes Log-Linear and an improved Log-Linear model.
*   **Evaluation:** Calculate perplexity and file log-probabilities.
*   **Text Generation:** Sample text from any trained model.

For more details, see the [`advanced-N-gram-and-log-linear-model/README.md`](advanced-N-gram-and-log-linear-model/README.md).

---

## 🌳 Context-Free Grammar (CFG)

A collection of scripts for experimenting with context-free grammars.

### Key Features:

*   **Sentence Generation:** Create random sentences based on a given grammar.
*   **Parsing:** Check if a sentence conforms to a specified grammar.
*   **Pretty Printing:** Format parsing results for better readability.

For more details, see the [`context-free-grammar/README.md`](context-free-grammar/README.md).

---

## 🔎 Cosine Similarity and Word Probability

This project provides tools to explore word similarity using pre-trained word embeddings.

### Key Features:

*   **Word Similarity:** Find the top-k most similar words to a given word.
*   **Word Analogies:** Perform vector arithmetic on word embeddings to solve analogies.
*   **Efficient:** Handles large embedding files effectively.

For more details, see the [`cosine_similarity_and_word_prob/README.md`](cosine_similarity_and_word_prob/README.md).

---

## 🛠️ How to Use

Each project in this playground is independent. To get started with a specific project, please navigate to its directory and consult the `README.md` file within it. You will find detailed instructions on dependencies, setup, and usage.

## 🤝 Contributing

Contributions are welcome! If you have an idea for a new project or an improvement to an existing one, please feel free to open an issue or submit a pull request.
