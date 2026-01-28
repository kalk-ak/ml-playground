#!/usr/bin/env python3

from __future__ import annotations
from optparse import Option
from pandas.core.computation.ops import Op
import beartype

import logging
import math
import pickle
import sys
import random

from pathlib import Path
from abc import abstractmethod

import torch
from torch import (
    Tensor,
    nn,
)
from torch import optim
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from beartype import beartype
from typing import Collection, Dict
from collections import Counter
from tqdm import tqdm


##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Tuple, Union
from integerize import Integerizer


Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab = Collection[Wordtype]  # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram = Tuple[Wordtype]
Bigram = Tuple[Wordtype, Wordtype]
Trigram = Tuple[Wordtype, Wordtype, Wordtype]
Ngram = Union[Zerogram, Unigram, Bigram, Trigram]
Vector = List[float]
TorchScalar = Float[
    torch.Tensor, ""
]  # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words

# Used for tensor strong typing
DIM: str = "dimension"
BATCH = "batch"


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

log: logging.Logger = logging.getLogger(
    Path(__file__).stem
)  # For usage, see findsim.py in earlier assignment.


def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(
    file: Path, vocab: Vocab, randomize: bool = False
) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for
    SGD training.

    If randomize is True, then randomize the order of the trigrams each time.
    This is more in the spirit of SGD, but the randomness makes the code harder to debug,
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools

        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random

        pool = tuple(trigrams)
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram


##### READ IN A VOCABULARY
def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    return sorted(vocab)


##### LANGUAGE MODEL PARENT CLASS


class LanguageModel:
    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0  # To print progress.

        self.event_count: Counter[Ngram] = Counter()
        self.context_count: Counter[Ngram] = Counter()
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z)] += 1
        self.event_count[(y, z)] += 1
        self.event_count[(z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion,
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram  # we don't care about z
        self.context_count[(x, y)] += 1
        self.context_count[(y,)] += 1
        self.context_count[()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly,
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError(
                "You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses."
            )
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = "cpu") -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(
                f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}"
            )
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")

    @abstractmethod
    def sample(self, n: int, max_len: int, device: str) -> str:
        """
        Samples n sentences from the language model. Must be implemented by subclasses

        arags:
            n: The number of sentences to generate

        Return Value:
            The generated sentence in string format
        """

        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError(
                "You shouldn't be calling sample on an instance of LanguageModel, but on an instance of one of its subclasses."
            )

        raise NotImplementedError(
            f"{class_name}.sample is not implemented yet (you should override LanguageModel.log_prob)"
        )


##### SPECIFIC FAMILIES OF LANGUAGE MODELS


class CountBasedLanguageModel(LanguageModel):
    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError(
                "You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses."
            )
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

    def sample(self, n: int = 1, max_len: int = 20, device: str = "cpu") -> str:
        assert isinstance(self.vocab, List), (
            f"Vocabulary is not a List. Vocab is of type {type(self.vocab)}"
        )

        vocab_list: List[Wordtype] = list(
            self.vocab
        )  # Convert the vocabulary to a list of words

        generated_sentences: List[str] = (
            [""] * n
        )  # Store the generated_sentences as a list because string is immutable and in-efficient

        # generate n sentences
        for i in range(n):
            # Initially every wordtype in the trigram is set to BOS
            x: Wordtype = BOS
            y: Wordtype = BOS

            current_sentence: List[Wordtype] = []
            for _ in range(max_len):
                vocab_log_probs: List[float] = [
                    self.log_prob(x, y, token) for token in vocab_list
                ]  # calculate the log probs for every token in the vocabulary

                # Convert the log prob into probability by exponentiating
                log_prob_tensor: Float[torch.Tensor, "Vocab"] = torch.tensor(
                    vocab_log_probs, dtype=torch.float32, device=device
                )

                weights: Float[torch.Tensor, "Vocab"] = torch.nn.functional.softmax(
                    log_prob_tensor, dim=0
                )

                index: int = int(torch.multinomial(weights, 1).item())
                choice: Wordtype = vocab_list[index]

                # Sample from our vocab and update x and y
                x, y = y, choice

                current_sentence.append(choice)

                if choice == EOS:
                    break

            # Store the generated_sentence and exclude EOS
            if current_sentence[-1] == EOS:
                current_sentence = current_sentence[:-1]
            else:
                current_sentence.append("...")  # Indicate continuation

            generated_sentences[i] = " ".join(current_sentence)

        return "\n".join(generated_sentences)


class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(
                f"Negative lambda argument of {lambda_} could result in negative smoothed probs"
            )
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return (self.event_count[x, y, z] + self.lambda_) / (
            self.context_count[x, y] + self.lambda_ * self.vocab_size
        )


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)
        self.lambda_scaled_vocab_ = self.lambda_ * self.vocab_size
        logging.debug(f"Vocab size = {self.vocab_size}")
        logging.debug(f"Vocab * Lamda = {self.lambda_scaled_vocab_}")

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return self.trigram_backoff((x, y, z))
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!

    def trigram_backoff(self, context: Trigram) -> float:
        """
        Computes the trigram probability. Recursively Backsoff to the bigram context

        args:
            context: Trigram tuple to get the count

        Return Value:
            float: the computed trigram probability
        """
        (x, y, z) = context
        return (
            self.event_count[x, y, z]
            + (self.lambda_scaled_vocab_ * self.bigram_backoff(context[1:]))
        ) / (self.event_count[x, y] + self.lambda_scaled_vocab_)

    def bigram_backoff(self, context: Bigram) -> float:
        """
        Computes the bigram probability. Recursively backsoff to unigram

        args:
            context: Bigram tuple to get the counts

        Return Value:
            float" The Computed Bigram probability
        """

        (y, z) = context

        return (
            self.event_count[y, z]
            + (self.lambda_scaled_vocab_ * self.unigram_backoff(context[1:]))
        ) / (self.event_count[(y,)] + self.lambda_scaled_vocab_)

    def unigram_backoff(self, context: Unigram) -> float:
        """
        # Computes the unigram probability. Recursively backsoff to Uniform LanguageModel

        args:
            context: Unigram tuple

        Return Value:
            float: The computed backoff unigram probability
        """

        return (self.event_count[context] + self.lambda_) / (
            self.event_count[()] + self.lambda_scaled_vocab_
        )


@jaxtyped(typechecker=beartype)  # pyright: ignore
class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.

    def __init__(
        self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int
    ) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError(f"Negative regularization strength {l2}")

        self.l2: float = l2
        self.epochs: int = (
            epochs  # Store the maximum number of epochs to trian this log linear model
        )

        assert BOS in self.vocab, "BOS is not part of vocab"
        logging.info("BOS is part of the vocab")

        ### START READING The FILE AND CREATING A TORCH to represent the lexicon file
        words: List[str] = []
        vectors: List[List[float]] = []
        # open the lexicon_file
        with open(lexicon_file, "r") as file:
            dimension: List[str] = file.readline().strip().split()
            row: int = int(dimension[0])
            col: int = int(dimension[1])

            # Index of the file line number
            i: int = 1

            # Stores the number of skipped_words from the lexicon due
            # either not being part of the vocabluary (backed-off to OOL) or
            # dimension is corrupted
            skipped_words: int = 0

            for line in file:
                parts: List[str] = line.strip().split()

                word: str = parts[0]
                vector: List[float] = [float(num) for num in parts[1:]]

                # Check if the dimension of the current lexicon is corrupted
                if len(vector) == col:
                    if word in self.vocab or word == OOL:
                        words.append(word)
                        vectors.append(vector)

                    else:
                        skipped_words += 1
                        logging.debug(
                            f"{word} at line {i} is not part of the Vocab. Replacing with OOV"
                        )
                    # else word will be considered as OOV
                else:
                    skipped_words += 1
                    logging.warning(
                        f"Skipping word found at line {i} because of dimension mismatch (corrupted dimension of {len(vector)}). Continuing ..."
                    )

                i += 1

            print(f"dimension = {dimension}\nrow = {row}\ncol = {col}")

        self.lexicon_matrice: Float[torch.Tensor, "Vocab DIM"] = torch.tensor(vectors)

        logging.info(f"Created a lexion matrix of shape {self.lexicon_matrice.shape}")
        logging.debug(f"Skipped {skipped_words} words")

        self.integerizer: Integerizer = Integerizer(words)
        logging.debug(
            f"Initialized Integerizer of length {len(self.integerizer)} successfully"
        )

        # Getting constant representations for the OOL embedding and caching it for future use
        self.OOL_INDEX: Optional[int] = self.integerizer.index(OOL)
        assert self.OOL_INDEX is not None, "OOL embedding doesn't exist in the lexicon"
        self.OOL_EMBEDDING: Float[torch.Tensor, "dimension"] = self.lexicon_matrice[
            self.OOL_INDEX
        ]

        vocab_vectors: List[Float[torch.Tensor, "DIM"]] = []
        token: Wordtype
        for token in vocab:
            # check if the token is part of the lexicon
            idx: Optional[int] = self.integerizer.index(token)

            if idx is not None:
                vocab_vectors.append(self.lexicon_matrice[idx])
            else:
                vocab_vectors.append(self.OOL_EMBEDDING)

        self.vocab_matrice: Float[torch.Tensor, "Vocab DIM"] = torch.stack(
            vocab_vectors
        )
        logging.debug(f"Created a Vocab metrix of shape {self.vocab_matrice.shape}")
        ### END OF READING THE FILE AND CREATING A TOARCH MATRIX

        self.dim: int = col  # set the dimension of the word embeddings

        # THE PARAMETERS TO BE TRAINED
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim), requires_grad=True))
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim), requires_grad=True))

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    def _get_embedding(self, token: Wordtype) -> Float[torch.Tensor, "DIM"]:
        token_idx: Optional[int] = self.integerizer.index(token)

        if token_idx is None:
            return self.OOL_EMBEDDING

        return self.lexicon_matrice[token_idx]

    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""

        x_embedding: Float[torch.Tensor, "DIM"] = self._get_embedding(x)
        y_embedding: Float[torch.Tensor, "DIM"] = self._get_embedding(y)
        z_embedding: Float[torch.Tensor, "DIM"] = self._get_embedding(z)

        # representing operations in log space
        context: Float[torch.Tensor, "DIM"] = (x_embedding @ self.X) + (
            y_embedding @ self.Y
        )

        numerator_log: TorchScalar = context @ z_embedding

        denominator_log: TorchScalar = self.logits(x, y).logsumexp(0)

        return numerator_log - denominator_log

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor, "Vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you
        exponentiate and renormalize in order to get a probability distribution."""

        x_embedding: Float[torch.Tensor, "DIM"] = self._get_embedding(x)
        y_embedding: Float[torch.Tensor, "DIM"] = self._get_embedding(y)

        # GOING TO REPRESENT Operation in logs
        context: Float[torch.Tensor, "DIM"] = (x_embedding @ self.X) + (
            y_embedding @ self.Y
        )

        calculated_logits: Float[torch.Tensor, "Vocab"] = (
            self.vocab_matrice @ context
        )  # same as (context @ self.lexicon_matrice.T but faster)

        return calculated_logits

    def train(self, file: Path) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Optimization hyperparameters.
        eta0: float = 1e-5  # initial learning rate

        optimizer = optim.SGD(self.parameters(), lr=eta0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)

        N = num_tokens(file)
        log.info(f"Start optimizing on {N} training tokens...")

        logging.info(f"Training from {file}")
        t: int = 1
        for e in range(self.epochs):
            logging.info(f"...Starting epoch {e}")

            total_loss: float = 0.0
            # Iterate over all the trigram in the given file
            for x, y, z in tqdm(read_trigrams(file, self.vocab), total=N):
                # Empty's accumulated grad for this computation
                optimizer.zero_grad()

                # decrease the learning rate
                eta: float = eta0 / (1 + eta0 * (2 * self.l2 / N) * t)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = eta

                self.eta0 = eta
                t += 1

                # Calculate the log prob for the current trigram
                actual_log_probability: TorchScalar = self.log_prob_tensor(x, y, z)

                regularizer: TorchScalar = (
                    self.l2 / N * (self.X.pow(2).sum() + self.Y.pow(2).sum())
                )

                # Since we are trying to maximize log_prob(x, y, z) - regularizer
                # we are going to minimizer regularizer - log_prob(x, y, z)
                objective_function: TorchScalar = actual_log_probability - regularizer
                loss: TorchScalar = -objective_function

                # Acumulate total loss for logging the average loss
                total_loss += loss.item()

                # Get the gradient with regards to the final loss
                loss.backward()

                # Update the weights of the matrix
                optimizer.step()

                # Show the current progress
                self.show_progress()

            logging.info(f"Epoch {e}: F = {-total_loss / N}.")

        log.info("done optimizing.")

    def sample(self, n: int, max_len: int, device: str) -> str:
        """
        Samples n sentences from the log-linear model using the trained parameters.
        """

        # We need to map indices back to words
        # self.vocab is a list, so self.vocab[i] gives the word at index i
        vocab_list: List[Wordtype] = list(self.vocab)
        generated_sentences: List[str] = []

        with torch.no_grad():  # No need to track gradients during sampling
            for _ in range(n):
                x, y = BOS, BOS
                sentence: List[str] = []

                for _ in range(max_len):
                    # Compute logits for the entire vocabulary given context (x, y)
                    next_word_logits: Float[torch.Tensor, "Vocab"] = self.logits(x, y)

                    # Apply Softmax to get Probabilities
                    probs = torch.nn.functional.softmax(next_word_logits, dim=0)

                    # Sample from the distribution
                    # num_samples=1 returns a tensor of one index
                    next_word_idx: int = torch.multinomial(probs, num_samples=1).item()  # ty: ignore
                    next_word: str = vocab_list[next_word_idx]

                    sentence.append(next_word)

                    if next_word == EOS:
                        break

                    # Update context
                    x, y = y, next_word

                # Format the string (remove EOS for cleanliness, handle truncation)
                if sentence and sentence[-1] == EOS:
                    sentence.pop()
                else:
                    sentence.append("...")  # Truncated

                generated_sentences.append(" ".join(sentence))

        return "\n".join(generated_sentences)


@jaxtyped(typechecker=beartype)  # pyright: ignore
class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):  # pyright: ignore
    def __init__(
        self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int
    ) -> None:
        # initialize to base class
        super().__init__(vocab, lexicon_file, l2, epochs)

        #  Normalize the full lexicon (used by _get_embedding during testing)
        # p=2 means L2 norm (Euclidean distance). dim=1 means normalize across columns.
        self.lexicon_matrice.data = torch.nn.functional.normalize(
            self.lexicon_matrice.data, p=2, dim=1
        )

        # Normalize the vocab matrix
        self.vocab_matrice.data = torch.nn.functional.normalize(
            self.vocab_matrice.data, p=2, dim=1
        )

        #  Update the OOL embedding cache to match the normalized version
        if self.OOL_INDEX is not None:
            self.OOL_EMBEDDING = self.lexicon_matrice[self.OOL_INDEX]

        # Initialize X and Y with a normal distribution (mean=0, std=0.01)
        # This breaks symmetry and helps with convergence
        nn.init.normal_(self.X, mean=0.0, std=0.01)
        nn.init.normal_(self.Y, mean=0.0, std=0.01)

    def train(self, file: Path) -> None:
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.lexicon_matrice = self.lexicon_matrice.to(device)
        self.vocab_matrice = self.vocab_matrice.to(device)
        logging.info(f"Training on {device}...")

        learning_rate: float = 0.05
        logging.info(f"Set learning rate to {learning_rate}")

        # --- DATA LOADING: Vectorization ---
        # Convert the entire training file into integer indices (0 to V-1)
        logging.info("Vectorizing training corpus...")

        # List of all trigrams
        trigrams: List[List[int]] = []

        # We map words to their index in the *Vocabulary* list, not the Lexicon.
        vocab_dict: Dict[Wordtype, int] = {w: i for i, w in enumerate(self.vocab)}

        for x, y, z in read_trigrams(file, self.vocab):
            trigrams.append([vocab_dict[x], vocab_dict[y], vocab_dict[z]])

        # Manually shuffle trigrams because data loader shuffles on the cpu
        # causing error in cuda
        random.shuffle(trigrams)

        # Create dataset and dataloader for mini-batching
        # have the data on the cpu and move the batches to the gpu
        data_tensor = torch.tensor(trigrams, dtype=torch.long, device=device)

        batch_size = 4096  # Adjust based on GPU memory
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        # Use Adam optimizer for faster convergence than standard SGD
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        N = len(trigrams)

        logging.info(
            f"Start optimizing on {N} training tokens ({self.epochs} epochs)..."
        )

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
                batch_idxs = batch[0]

                # split into x, y, z indices (Vocab Indices)
                x_idx, y_idx, z_idx = (
                    batch_idxs[:, 0],
                    batch_idxs[:, 1],
                    batch_idxs[:, 2],
                )

                # Lookup Embeddings for Context (x, y)
                # Map Vocab Indices -> Lexicon Indices -> Embeddings
                x_emb: Float[torch.Tensor, "BATCH DIM"] = self.vocab_matrice[
                    x_idx
                ]  # (Batch, Dim)

                y_emb: Float[torch.Tensor, "BATCH DIM"] = self.vocab_matrice[
                    y_idx
                ]  # (Batch, Dim)

                # Compute Context Vector
                # context = xX + yY
                context: Float[torch.Tensor, "BATCH DIM"] = (x_emb @ self.X) + (
                    y_emb @ self.Y
                )  # (Batch, Dim)

                # Compute Logits for ALL words in Vocab
                # (Batch, Dim) @ (Vocab, Dim).T -> (Batch, Vocab)
                # We construct the full vocab matrix dynamically to ensure gradients flow
                logits: Float[torch.Tensor, "BATCH Vocab"] = (
                    context @ self.vocab_matrice.T
                )

                # log_prob = logit_target - logsumexp(all_logits)
                # Get the score for the correct target 'z'
                target_scores: Float[torch.Tensor, "BATCH"] = logits.gather(
                    1, z_idx.unsqueeze(1)
                ).squeeze()

                # Compute Z (normalization constant) using logsumexp
                log_Z = torch.logsumexp(logits, dim=1)

                # Negative Log Likelihood
                # We minimize negative log prob, so loss = - (score - logZ)
                loss = -((target_scores - log_Z).mean())

                # Add L2 Regularization
                l2_reg = (self.l2 / N) * (self.X.pow(2).sum() + self.Y.pow(2).sum())
                loss += l2_reg

                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_idxs)

            avg_loss = total_loss / N
            logging.info(
                f"Epoch {epoch + 1}: F = {-avg_loss}"
            )  # Print negative loss (F)
