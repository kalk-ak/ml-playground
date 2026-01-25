# Trigram Language Modeling & Log-Linear Models

A comprehensive Python toolkit for training, evaluating, and sampling from N-gram language models. This project implements both classic statistical smoothing techniques and modern Neural Log-Linear models using PyTorch. Code is written from the ground up to use highly optimized pytorch and efficient vectorizations.

## 🚀 Features

- **Statistical Models**:
  - **Uniform**: Baseline model with equal probability for all tokens.
  - **Add-Lambda**: Laplace smoothing to handle zero-count N-grams.
  - **Backoff Add-Lambda**: Recursive backoff (Trigram $\to$ Bigram $\to$ Unigram) for better generalization.
- **Neural Models**:
  - **Log-Linear**: Embedding-based language model trained with SGD/Adam.
  - **Improved Log-Linear**: Enhanced version with normalized embeddings and better initialization.
- **Evaluation Tools**: Calculate perplexity (cross-entropy) and file log-probabilities.
- **Sampling**: Generate text from any trained model.
- **Efficient**: Uses `torch` for tensor operations and `tqdm` for progress tracking.

## 📂 Project Structure

- `probs.py`: Core library defining the `LanguageModel` abstract base class and all implementations.
- `train_lm.py`: Main CLI entry point for training models.
- `fileprob.py`: Tool to evaluate models on test data (calculates cross-entropy/bits per token).
- `build_vocab.py`: Utility to generate vocabulary files from corpora.
- `combine_vocab.py`: Helper to merge vocabulary files.
- `train_log_linear.ipynb`: Jupyter Notebook optimized for training neural models on Google Colab (GPU).

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   _Requires Python 3.9+_

## 📖 Usage

All scripts in this project include a comprehensive help interface. You can use the `-h` or `--help` flag with any script to see all available options and default values.

```bash
python train_lm.py -h
python build_vocab.py -h
python fileprob.py -h
```

### 1. Building a Vocabulary

Before training, generate a vocabulary file from your training corpus.

```bash
# Create a vocab from two files with a minimum count threshold of 3
python build_vocab.py data/train/gen data/train/spam --threshold 3 --output vocab.txt
```

### 2. Training Models

Use `train_lm.py` to train models. The script saves the model as a `.model` file (serialized Python object).

**Syntax:**

```bash
python train_lm.py <vocab_file> <smoother_type> <train_file> [options]
```

#### Statistical Models

```bash
# Uniform Model
python train_lm.py vocab.txt uniform data/train/corpus.txt

# Add-Lambda (Laplace) Smoothing
python train_lm.py vocab.txt add_lambda data/train/corpus.txt --lambda 0.1

# Backoff with Add-Lambda
python train_lm.py vocab.txt add_lambda_backoff data/train/corpus.txt --lambda 0.1
```

#### Neural Log-Linear Models

Requires a lexicon file (word embeddings).

```bash
# Standard Log-Linear
python train_lm.py vocab.txt log_linear data/train/corpus.txt \
  --lexicon data/lexicons/glove.txt \
  --epochs 10 \
  --l2_regularization 0.01

# Improved Log-Linear (Recommended for GPU)
python train_lm.py vocab.txt log_linear_improved data/train/corpus.txt \
  --lexicon data/lexicons/glove.txt \
  --epochs 20 \
  --l2_regularization 0.5 \
  --device cuda
```

### 3. Evaluation (Perplexity)

Calculate the cross-entropy (bits per token) of a trained model on a test set.

```bash
python fileprob.py <model_file> <test_file1> [test_file2 ...]
```

**Example Output:**

```text
-450.23 data/test/email1.txt
Overall cross-entropy: 5.43210 bits per token
```

### 4. Text Generation (Sampling)

You can sample sentences from any trained model class programmatically or by extending the scripts.
(See `LanguageModel.sample` method in `probs.py`).

## 🧠 Model Details

### Log-Linear Model

The log-linear model predicts $P(z | x, y)$ using word embeddings:

$$
\\
\text{score}(z | x, y) = (E_x X + E_y Y) \cdot E_z
$$

Where:

- $E_w$ is the embedding for word $w$.
- $X, Y$ are learned weight matrices transforming the context embeddings.

### Backoff Smoothing

Approximates probabilities by interpolating lower-order N-grams when higher-order counts are sparse:

$$
\\
P_{smooth}(z|x,y) \approx \frac{C(x,y,z) + \alpha P_{backoff}(z|y)}{C(x,y) + \alpha}
$$

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-smoother`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.
