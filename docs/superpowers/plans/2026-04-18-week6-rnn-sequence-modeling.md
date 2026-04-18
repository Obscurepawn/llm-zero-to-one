# Week 6 RNN Sequence Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the week 6 RNN experiment package with runnable code, aligned docs, and generated visuals that teach character-level RNNs, hidden-state dynamics, and long-sequence training difficulty.

**Architecture:** Add a new `experiments/rnn-and-sequence-modeling/` directory containing a main character-level RNN script built around `nn.RNNCell`, a separate sequence-length probe script, a unified `run_all.py`, week-specific documentation, and an SVG computation graph. Keep core helpers importable so lightweight `unittest` coverage can validate tokenizer logic, dataset construction, forward shapes, probe summaries, and output generation without relying on long training jobs.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, standard-library `unittest`, SVG, Markdown

---

## File Structure

- Create: `experiments/rnn-and-sequence-modeling/char_rnn_next_char.py`
  - Week 6 main experiment: corpus, tokenizer, dataset builder, `CharRNNModel`, training loop, sampling, prediction trace, plots, `main()`
- Create: `experiments/rnn-and-sequence-modeling/rnn_sequence_length_probe.py`
  - Compares multiple context lengths and plots training difficulty metrics
- Create: `experiments/rnn-and-sequence-modeling/run_all.py`
  - Runs the main experiment first, then the probe script
- Create: `experiments/rnn-and-sequence-modeling/README.md`
  - Week 6 learner-facing guide aligned with earlier weeks
- Create: `experiments/rnn-and-sequence-modeling/rnn_advantages_and_limits.md`
  - One-page concept note on strengths, weaknesses, and the bridge to LSTM/GRU
- Create: `experiments/rnn-and-sequence-modeling/rnn_unrolled_computation_graph.svg`
  - Visualizes `x_t`, `e_t`, `h_t`, logits, shared weights, and BPTT arrows
- Create: `experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py`
  - Unit tests for corpus construction, tokenizer round-trip, dataset shapes, forward pass, sampling, and plotting helpers
- Create: `experiments/rnn-and-sequence-modeling/tests/test_rnn_sequence_length_probe.py`
  - Unit tests for probe summaries and the comparison plot
- Modify: `notes/transformer-study-path.md`
  - Replace the generic week 6 “代码实践” bullets with concrete file references, matching weeks 2-4

### Task 1: Scaffold the Week 6 Data Pipeline

**Files:**
- Create: `experiments/rnn-and-sequence-modeling/char_rnn_next_char.py`
- Create: `experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py`

- [ ] **Step 1: Write the failing tests for the corpus, tokenizer, and dataset helpers**

```python
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from char_rnn_next_char import CharTokenizer, build_demo_corpus, build_sequence_dataset


class CharRNNBasicsTest(unittest.TestCase):
    def test_demo_corpus_mentions_recurrent_concepts(self) -> None:
        corpus = build_demo_corpus(repeats=2)
        self.assertIn("hidden states", corpus)
        self.assertIn("time steps", corpus)
        self.assertGreater(len(corpus), 120)

    def test_tokenizer_roundtrip_and_dataset_shapes(self) -> None:
        tokenizer = CharTokenizer("abba\n")
        token_ids = tokenizer.encode("abbaabba")

        self.assertEqual(tokenizer.decode(token_ids[:4]), "abba")
        self.assertGreater(tokenizer.vocab_size, 2)

        inputs, targets = build_sequence_dataset(token_ids, context_length=4)
        self.assertEqual(inputs.shape, (4, 4))
        self.assertEqual(targets.shape, (4, 4))
        self.assertEqual(inputs[0].tolist(), token_ids[:4])
        self.assertEqual(targets[0].tolist(), token_ids[1:5])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail before implementation**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_char_rnn_next_char.py" -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `char_rnn_next_char.py` does not exist yet.

- [ ] **Step 3: Create the initial helper implementation in `char_rnn_next_char.py`**

```python
#!/usr/bin/env python3
"""Week 6: character-level RNN and sequence modeling basics."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_demo_corpus(repeats: int = 6) -> str:
    lines = [
        "recurrent models read one token at a time.",
        "hidden states carry context across time steps.",
        "simple rnn models share parameters at every step.",
        "backpropagation through time gets harder on long sequences.",
    ]
    return "\n".join(lines * repeats)


class CharTokenizer:
    def __init__(self, text: str, special_tokens: tuple[str, ...] = ("<PAD>", "<UNK>")) -> None:
        unique_tokens = sorted(set(text))
        self.id_to_token = list(special_tokens) + unique_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.special_tokens = set(special_tokens)
        self.special_token_ids = {self.token_to_id[token] for token in special_tokens}
        self.pad_token = special_tokens[0]
        self.unk_token = special_tokens[1]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(char, unk_id) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in {self.pad_token, self.unk_token}:
                continue
            tokens.append(token)
        return "".join(tokens)


def build_sequence_dataset(token_ids: list[int], context_length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(token_ids) <= context_length:
        raise ValueError("token_ids must be longer than context_length")

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(len(token_ids) - context_length):
        window = token_ids[start : start + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[1:])

    return np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)
```

- [ ] **Step 4: Run the tests again to verify the helpers pass**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_char_rnn_next_char.py" -v`

Expected: PASS with 2 tests passing.

- [ ] **Step 5: Commit the scaffold**

```bash
git add experiments/rnn-and-sequence-modeling/char_rnn_next_char.py \
        experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py
git commit -m "feat: scaffold week 6 rnn data pipeline"
```

### Task 2: Add the RNN Model and Forward Sequence Logic

**Files:**
- Modify: `experiments/rnn-and-sequence-modeling/char_rnn_next_char.py`
- Modify: `experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py`

- [ ] **Step 1: Extend the tests with a failing forward-pass shape check**

```python
import torch

from char_rnn_next_char import CharRNNModel


class CharRNNForwardTest(unittest.TestCase):
    def test_forward_sequence_returns_logits_and_hidden_states(self) -> None:
        torch.manual_seed(0)
        model = CharRNNModel(vocab_size=9, embedding_dim=6, hidden_dim=5)
        batch = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)

        logits, hidden_states = model.forward_sequence(batch)

        self.assertEqual(tuple(logits.shape), (2, 4, 9))
        self.assertEqual(tuple(hidden_states.shape), (2, 4, 5))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(hidden_states).all())
```

- [ ] **Step 2: Run the test file and confirm it fails because `CharRNNModel` is missing**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_char_rnn_next_char.py" -v`

Expected: FAIL with `NameError: name 'CharRNNModel' is not defined`.

- [ ] **Step 3: Implement `CharRNNModel` and the explicit time-step loop**

```python
import torch.nn as nn


class CharRNNModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 24, hidden_dim: int = 48) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_cell = nn.RNNCell(input_size=embedding_dim, hidden_size=hidden_dim, nonlinearity="tanh")
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_sequence(
        self,
        token_batch: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = token_batch.shape
        if hidden is None:
            hidden = self.init_hidden(batch_size, token_batch.device)

        embeddings = self.embedding(token_batch)
        logits_steps: list[torch.Tensor] = []
        hidden_steps: list[torch.Tensor] = []

        for step_idx in range(seq_len):
            hidden = self.rnn_cell(embeddings[:, step_idx, :], hidden)
            hidden_steps.append(hidden)
            logits_steps.append(self.output_head(hidden))

        logits = torch.stack(logits_steps, dim=1)
        hidden_states = torch.stack(hidden_steps, dim=1)
        return logits, hidden_states

    def forward(self, token_batch: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_sequence(token_batch)
        return logits
```

- [ ] **Step 4: Re-run the tests and verify the forward logic passes**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_char_rnn_next_char.py" -v`

Expected: PASS with 3 tests passing.

- [ ] **Step 5: Commit the model**

```bash
git add experiments/rnn-and-sequence-modeling/char_rnn_next_char.py \
        experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py
git commit -m "feat: add week 6 char rnn forward pass"
```

### Task 3: Implement Training, Sampling, Predictions, and Main Visuals

**Files:**
- Modify: `experiments/rnn-and-sequence-modeling/char_rnn_next_char.py`
- Modify: `experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py`

- [ ] **Step 1: Add failing smoke tests for training, sampling, and plot generation**

```python
from tempfile import TemporaryDirectory

from char_rnn_next_char import (
    CharRNNModel,
    CharTokenizer,
    build_demo_corpus,
    build_sequence_dataset,
    plot_hidden_state_dynamics,
    plot_prediction_trace,
    plot_training_curves,
    predict_next_chars,
    sample_text,
    train_model,
)


class CharRNNTrainingSmokeTest(unittest.TestCase):
    def test_training_history_sampling_and_plots(self) -> None:
        torch.manual_seed(0)

        corpus = build_demo_corpus(repeats=1)
        tokenizer = CharTokenizer(corpus)
        token_ids = tokenizer.encode(corpus)
        inputs, targets = build_sequence_dataset(token_ids[:80], context_length=8)

        model = CharRNNModel(vocab_size=tokenizer.vocab_size, embedding_dim=12, hidden_dim=16)
        history = train_model(
            model,
            inputs,
            targets,
            num_epochs=2,
            batch_size=4,
            learning_rate=0.03,
            clip_grad=1.0,
        )

        self.assertEqual(len(history["loss"]), 2)
        self.assertEqual(len(history["accuracy"]), 2)
        self.assertEqual(len(history["grad_norm"]), 2)

        sample = sample_text(
            model,
            tokenizer,
            seed_text="recur",
            context_length=8,
            sample_length=12,
            temperature=1.0,
        )
        self.assertEqual(len(sample), len("recur") + 12)

        predictions = predict_next_chars(
            model,
            tokenizer,
            prompt="hidden",
            context_length=8,
            top_k=3,
        )
        self.assertEqual(len(predictions), 3)

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            plot_training_curves(history, output_dir / "char_rnn_training_curves.png")
            plot_prediction_trace({"hidden": predictions}, output_dir / "rnn_prediction_trace.png")
            plot_hidden_state_dynamics(
                model,
                tokenizer,
                text="hidden states carry context",
                context_length=8,
                output_path=output_dir / "rnn_hidden_state_dynamics.png",
            )
            self.assertTrue((output_dir / "char_rnn_training_curves.png").exists())
            self.assertTrue((output_dir / "rnn_prediction_trace.png").exists())
            self.assertTrue((output_dir / "rnn_hidden_state_dynamics.png").exists())
```

- [ ] **Step 2: Run the tests and confirm they fail because the helpers are missing**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_char_rnn_next_char.py" -v`

Expected: FAIL with `NameError` for `train_model`, `sample_text`, `predict_next_chars`, or plotting helpers.

- [ ] **Step 3: Implement training, next-character predictions, and text sampling**

```python
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model: CharRNNModel,
    inputs: np.ndarray,
    targets: np.ndarray,
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.02,
    clip_grad: float = 1.0,
) -> dict[str, list[float]]:
    dataset = TensorDataset(
        torch.from_numpy(inputs).long(),
        torch.from_numpy(targets).long(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": [], "grad_norm": []}

    model.train()
    for _ in range(num_epochs):
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_grad_norm = 0.0

        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            logits, _ = model.forward_sequence(batch_inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            total_loss += loss.item() * batch_targets.numel()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == batch_targets).sum().item()
            total_tokens += batch_targets.numel()
            total_grad_norm += float(grad_norm)

        history["loss"].append(total_loss / total_tokens)
        history["accuracy"].append(total_correct / total_tokens)
        history["grad_norm"].append(total_grad_norm / len(dataloader))

    return history


def predict_next_chars(
    model: CharRNNModel,
    tokenizer: CharTokenizer,
    prompt: str,
    context_length: int,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    model.eval()
    encoded = tokenizer.encode(prompt)[-context_length:]
    padded = [tokenizer.token_to_id[tokenizer.pad_token]] * max(0, context_length - len(encoded)) + encoded
    batch = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logits, _ = model.forward_sequence(batch)
        next_logits = logits[0, -1].clone()
        for special_id in tokenizer.special_token_ids:
            next_logits[special_id] = -1e9
        probabilities = torch.softmax(next_logits, dim=-1)
        values, indices = torch.topk(probabilities, k=top_k)

    return [
        (tokenizer.id_to_token[index.item()], float(value.item()))
        for value, index in zip(values, indices)
    ]


def sample_text(
    model: CharRNNModel,
    tokenizer: CharTokenizer,
    seed_text: str,
    context_length: int,
    sample_length: int = 80,
    temperature: float = 1.0,
) -> str:
    generated_token_ids = tokenizer.encode(seed_text)
    model.eval()

    for _ in range(sample_length):
        encoded = generated_token_ids[-context_length:]
        padded = [tokenizer.token_to_id[tokenizer.pad_token]] * max(0, context_length - len(encoded)) + encoded
        batch = torch.tensor([padded], dtype=torch.long)

        with torch.no_grad():
            logits, _ = model.forward_sequence(batch)
            scaled_logits = logits[0, -1].clone() / max(temperature, 1e-6)
            for special_id in tokenizer.special_token_ids:
                scaled_logits[special_id] = -1e9
            probabilities = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

        generated_token_ids.append(next_token)

    return tokenizer.decode(generated_token_ids)
```

- [ ] **Step 4: Implement the plotters, text-output writer, and `main()`**

```python
import matplotlib.pyplot as plt


def plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["loss"], label="loss", color="#1f77b4")
    axes[0].plot(epochs, history["accuracy"], label="accuracy", color="#ff7f0e")
    axes[0].set_title("Week 6 Char RNN Training")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["grad_norm"], color="#2ca02c")
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_hidden_state_dynamics(
    model: CharRNNModel,
    tokenizer: CharTokenizer,
    text: str,
    context_length: int,
    output_path: Path,
) -> None:
    encoded = tokenizer.encode(text[:context_length])
    batch = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        _, hidden_states = model.forward_sequence(batch)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(hidden_states[0].T.numpy(), aspect="auto", cmap="coolwarm")
    ax.set_title("Hidden State Dynamics Across Time Steps")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Hidden Dimension")
    ax.set_xticks(range(len(encoded)))
    ax.set_xticklabels([tokenizer.id_to_token[token_id] for token_id in encoded], rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prediction_trace(
    predictions_by_prompt: dict[str, list[tuple[str, float]]],
    output_path: Path,
) -> None:
    prompts = list(predictions_by_prompt.keys())
    fig, axes = plt.subplots(len(prompts), 1, figsize=(10, 3 * len(prompts)))
    axes = np.atleast_1d(axes)

    for ax, prompt in zip(axes, prompts):
        labels = [token for token, _ in predictions_by_prompt[prompt]]
        values = [score for _, score in predictions_by_prompt[prompt]]
        ax.bar(labels, values, color="#4c78a8")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Top-k next-char predictions for '{prompt}'")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_generated_samples(samples: dict[float, str], output_path: Path) -> None:
    lines = []
    for temperature, sample in samples.items():
        lines.append(f"[temperature={temperature}]")
        lines.append(sample)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(42)
    output_dir = ensure_output_dir()
    corpus = build_demo_corpus()
    tokenizer = CharTokenizer(corpus)
    token_ids = tokenizer.encode(corpus)
    context_length = 16
    inputs, targets = build_sequence_dataset(token_ids, context_length=context_length)

    model = CharRNNModel(vocab_size=tokenizer.vocab_size, embedding_dim=24, hidden_dim=48)
    history = train_model(model, inputs, targets, num_epochs=20, batch_size=16, learning_rate=0.02)

    plot_training_curves(history, output_dir / "char_rnn_training_curves.png")
    plot_hidden_state_dynamics(
        model,
        tokenizer,
        text="hidden states carry context",
        context_length=context_length,
        output_path=output_dir / "rnn_hidden_state_dynamics.png",
    )

    prediction_trace = {
        "recur": predict_next_chars(model, tokenizer, "recur", context_length=context_length, top_k=5),
        "hidden": predict_next_chars(model, tokenizer, "hidden", context_length=context_length, top_k=5),
        "long ": predict_next_chars(model, tokenizer, "long ", context_length=context_length, top_k=5),
    }
    plot_prediction_trace(prediction_trace, output_dir / "rnn_prediction_trace.png")

    save_generated_samples(
        {
            0.8: sample_text(model, tokenizer, "recur", context_length=context_length, sample_length=80, temperature=0.8),
            1.0: sample_text(model, tokenizer, "hidden", context_length=context_length, sample_length=80, temperature=1.0),
            1.2: sample_text(model, tokenizer, "long ", context_length=context_length, sample_length=80, temperature=1.2),
        },
        output_dir / "rnn_generated_samples.txt",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Re-run the unit tests and verify the training smoke test passes**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_char_rnn_next_char.py" -v`

Expected: PASS with 4 tests passing.

- [ ] **Step 6: Run the main script once and verify the expected outputs are written**

Run: `python experiments/rnn-and-sequence-modeling/char_rnn_next_char.py`

Expected: The directory `experiments/rnn-and-sequence-modeling/outputs/` contains:
- `char_rnn_training_curves.png`
- `rnn_hidden_state_dynamics.png`
- `rnn_prediction_trace.png`
- `rnn_generated_samples.txt`

- [ ] **Step 7: Commit the runnable week 6 main experiment**

```bash
git add experiments/rnn-and-sequence-modeling/char_rnn_next_char.py \
        experiments/rnn-and-sequence-modeling/tests/test_char_rnn_next_char.py
git commit -m "feat: add week 6 char rnn experiment"
```

### Task 4: Add the Sequence-Length Probe and Unified Runner

**Files:**
- Create: `experiments/rnn-and-sequence-modeling/rnn_sequence_length_probe.py`
- Create: `experiments/rnn-and-sequence-modeling/run_all.py`
- Create: `experiments/rnn-and-sequence-modeling/tests/test_rnn_sequence_length_probe.py`

- [ ] **Step 1: Write the failing tests for the probe summary and plot**

```python
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnn_sequence_length_probe import plot_sequence_length_comparison, run_probe


class SequenceLengthProbeTest(unittest.TestCase):
    def test_run_probe_returns_one_summary_per_context_length(self) -> None:
        results = run_probe(context_lengths=(4, 8), num_epochs=1, repeats=1)
        self.assertEqual([item["context_length"] for item in results], [4, 8])
        self.assertEqual(len(results), 2)
        self.assertIn("final_loss", results[0])
        self.assertIn("max_grad_norm", results[0])

    def test_plot_sequence_length_comparison_creates_png(self) -> None:
        results = [
            {"context_length": 4, "final_loss": 1.9, "final_accuracy": 0.33, "max_grad_norm": 0.7},
            {"context_length": 8, "final_loss": 2.2, "final_accuracy": 0.24, "max_grad_norm": 1.1},
        ]
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rnn_sequence_length_comparison.png"
            plot_sequence_length_comparison(results, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the probe tests to verify they fail before implementation**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_rnn_sequence_length_probe.py" -v`

Expected: FAIL with `ModuleNotFoundError` because `rnn_sequence_length_probe.py` does not exist yet.

- [ ] **Step 3: Implement the probe script**

```python
#!/usr/bin/env python3
"""Compare how simple RNN training changes as sequence length grows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from char_rnn_next_char import (
    CharRNNModel,
    CharTokenizer,
    OUTPUT_DIR,
    build_demo_corpus,
    build_sequence_dataset,
    ensure_output_dir,
    set_seed,
    train_model,
)


def run_probe(
    context_lengths: tuple[int, ...] = (4, 8, 16, 24),
    num_epochs: int = 10,
    repeats: int = 4,
) -> list[dict[str, float]]:
    set_seed(7)
    corpus = build_demo_corpus(repeats=repeats)
    tokenizer = CharTokenizer(corpus)
    token_ids = tokenizer.encode(corpus)
    results: list[dict[str, float]] = []

    for context_length in context_lengths:
        trimmed = token_ids[: max(context_length + 40, 120)]
        inputs, targets = build_sequence_dataset(trimmed, context_length=context_length)
        model = CharRNNModel(vocab_size=tokenizer.vocab_size, embedding_dim=16, hidden_dim=32)
        history = train_model(
            model,
            inputs,
            targets,
            num_epochs=num_epochs,
            batch_size=8,
            learning_rate=0.025,
            clip_grad=1.0,
        )
        results.append(
            {
                "context_length": float(context_length),
                "final_loss": float(history["loss"][-1]),
                "final_accuracy": float(history["accuracy"][-1]),
                "max_grad_norm": float(max(history["grad_norm"])),
            }
        )

    return results


def plot_sequence_length_comparison(results: list[dict[str, float]], output_path: Path) -> None:
    context_lengths = [int(item["context_length"]) for item in results]
    losses = [item["final_loss"] for item in results]
    accuracies = [item["final_accuracy"] for item in results]
    grad_norms = [item["max_grad_norm"] for item in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(context_lengths, losses, marker="o", color="#1f77b4")
    axes[0].set_title("Final Loss vs Context Length")
    axes[0].set_xlabel("Context Length")

    axes[1].plot(context_lengths, accuracies, marker="o", color="#ff7f0e")
    axes[1].set_title("Final Accuracy vs Context Length")
    axes[1].set_xlabel("Context Length")

    axes[2].plot(context_lengths, grad_norms, marker="o", color="#2ca02c")
    axes[2].set_title("Max Gradient Norm vs Context Length")
    axes[2].set_xlabel("Context Length")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = ensure_output_dir(OUTPUT_DIR)
    results = run_probe()
    plot_sequence_length_comparison(results, output_dir / "rnn_sequence_length_comparison.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the probe tests again and verify they pass**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -p "test_rnn_sequence_length_probe.py" -v`

Expected: PASS with 2 tests passing.

- [ ] **Step 5: Add the unified `run_all.py` entrypoint**

```python
#!/usr/bin/env python3
"""Run all week 6 RNN experiments."""

from __future__ import annotations

import char_rnn_next_char
import rnn_sequence_length_probe


def main() -> None:
    print("Running week 6 char-level RNN experiment...\n")
    char_rnn_next_char.main()

    print("\n" + "=" * 80 + "\n")

    print("Running week 6 sequence-length probe...\n")
    rnn_sequence_length_probe.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit the probe and runner**

```bash
git add experiments/rnn-and-sequence-modeling/rnn_sequence_length_probe.py \
        experiments/rnn-and-sequence-modeling/run_all.py \
        experiments/rnn-and-sequence-modeling/tests/test_rnn_sequence_length_probe.py
git commit -m "feat: add week 6 sequence length probe"
```

### Task 5: Write the Learner-Facing Docs and Update the Study Path

**Files:**
- Create: `experiments/rnn-and-sequence-modeling/README.md`
- Create: `experiments/rnn-and-sequence-modeling/rnn_advantages_and_limits.md`
- Create: `experiments/rnn-and-sequence-modeling/rnn_unrolled_computation_graph.svg`
- Modify: `notes/transformer-study-path.md`

- [ ] **Step 1: Write the week 6 `README.md`**

```markdown
# 第 6 周：RNN 和序列建模的第一步

这个目录对应学习计划中的第 6 周，目标是把“token 进入 embedding 之后，模型如何按时间步处理序列”真正跑通。

前一周你已经看过：

- 文本如何变成 token id
- one-hot 和 embedding 的差别
- 语言模型任务是“根据前文预测下一个 token”

这一周继续往前走，重点变成：

- RNN 如何一边读入字符，一边更新 hidden state
- 为什么同一个 RNN 单元可以处理不同长度的序列
- 为什么序列一长，训练就更容易变难

## 目录结构

- `char_rnn_next_char.py`
  - Week 6 主实验
  - 用 `Embedding -> RNNCell -> Linear` 做字符级 next-character prediction
  - 输出训练曲线、hidden state 热力图、预测轨迹和生成样本
- `rnn_sequence_length_probe.py`
  - 对比不同序列长度下的训练难度
  - 输出 `rnn_sequence_length_comparison.png`
- `rnn_advantages_and_limits.md`
  - 总结 RNN 的优势和局限
- `rnn_unrolled_computation_graph.svg`
  - 解释时间展开、参数共享和 BPTT
- `run_all.py`
  - 本周统一入口
- `outputs/`
  - 运行后自动生成的图片和文本

## 为什么这里使用字符级 RNN

字符级任务足够简单，能把下面这条主线看得很清楚：

`字符 -> token id -> embedding -> hidden state -> logits -> 下一个字符`

如果这里一开始就上更复杂的词级模型，初学者更容易被词表、预处理和语料细节分散注意力。

## 为什么主实验使用 `RNNCell`

如果直接使用 `nn.RNN`，时间步会被封装进框架内部。对于教学来说，第 6 周更重要的是看清楚：

- 每一步输入了什么
- hidden state 是怎么更新的
- 同一组参数是如何在每个时间步重复使用的

所以这里故意使用 `RNNCell` 显式展开时间维。

## 如何运行

在当前目录执行：

```bash
python char_rnn_next_char.py
python rnn_sequence_length_probe.py
```

如果你想一次性跑完整周实验：

```bash
python run_all.py
```

## 运行后重点观察什么

1. `outputs/char_rnn_training_curves.png`
   - 看 loss 是否下降
   - 看 gradient norm 是否稳定
2. `outputs/rnn_hidden_state_dynamics.png`
   - 看 hidden state 怎样随着字符输入不断变化
3. `outputs/rnn_prediction_trace.png`
   - 看模型在不同上下文后的 top-k 下一个字符预测
4. `outputs/rnn_sequence_length_comparison.png`
   - 看序列变长后训练是否更难
5. `outputs/rnn_generated_samples.txt`
   - 看模型学到的局部字符模式和长距离局限

## 本周学完后，你至少应该能回答的问题

1. hidden state 在 RNN 里到底保存了什么？
2. 为什么同一个 RNN 单元可以处理变长序列？
3. BPTT 相比普通反向传播，多了哪一层沿时间展开的含义？
4. 为什么 vanilla RNN 难以处理很长距离的依赖？
```

- [ ] **Step 2: Write `rnn_advantages_and_limits.md`**

```markdown
# RNN 的优势和问题

## 为什么 RNN 很自然

RNN 的核心想法很直接：

- 每次读一个时间步输入
- 用上一时刻的 hidden state 和当前输入一起算出新的 hidden state
- 再用当前 hidden state 做预测

这种结构有三个明显优点：

1. 可以处理变长序列
2. 参数在所有时间步共享
3. 很符合“按顺序读入信息”的直觉

## 为什么 vanilla RNN 会变难

问题也同样明显：

1. 训练需要沿时间展开，BPTT 链条会越来越长
2. 梯度在长链条上传播时容易衰减或放大
3. 每个时间步都依赖前一步，天然不容易并行

这就会带来两个常见现象：

- 长距离依赖难学
- 序列越长，训练越容易不稳定或效果变差

## 这一周和下一周怎么衔接

第 6 周的目的不是证明 RNN 很完美，而是先把最基础的时序建模直觉建立起来。

当你真正看清楚 vanilla RNN 为什么会在长序列上吃力，第 7 周的 LSTM 和 GRU 才会显得自然：

- 它们不是凭空出现的新模块
- 它们是在解决 vanilla RNN 的长期依赖和梯度传递问题
```

- [ ] **Step 3: Write `rnn_unrolled_computation_graph.svg`**

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="980" height="420" viewBox="0 0 980 420">
  <style>
    .box { fill: #f7f7f2; stroke: #333; stroke-width: 2; rx: 12; ry: 12; }
    .state { fill: #dbeafe; stroke: #1d4ed8; stroke-width: 2; rx: 12; ry: 12; }
    .embed { fill: #dcfce7; stroke: #15803d; stroke-width: 2; rx: 12; ry: 12; }
    .logits { fill: #fee2e2; stroke: #b91c1c; stroke-width: 2; rx: 12; ry: 12; }
    .label { font: 16px sans-serif; fill: #111827; }
    .small { font: 13px sans-serif; fill: #374151; }
    .arrow { stroke: #111827; stroke-width: 2.2; fill: none; marker-end: url(#arrow); }
    .back { stroke: #7c3aed; stroke-width: 2.2; fill: none; stroke-dasharray: 8 6; marker-end: url(#arrow-purple); }
  </style>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#111827"/>
    </marker>
    <marker id="arrow-purple" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#7c3aed"/>
    </marker>
  </defs>

  <text x="30" y="36" class="label">Week 6: Unrolled RNN Computation Graph</text>
  <text x="30" y="62" class="small">Shared parameters: W_xh, W_hh, W_hy reused at every time step</text>

  <rect x="60" y="290" width="90" height="44" class="box"/>
  <text x="92" y="318" class="label">x₁</text>
  <rect x="180" y="290" width="110" height="44" class="embed"/>
  <text x="218" y="318" class="label">e₁</text>
  <rect x="320" y="200" width="130" height="60" class="state"/>
  <text x="357" y="236" class="label">h₁</text>
  <rect x="490" y="200" width="130" height="60" class="state"/>
  <text x="527" y="236" class="label">h₂</text>
  <rect x="660" y="200" width="130" height="60" class="state"/>
  <text x="697" y="236" class="label">h₃</text>

  <rect x="350" y="92" width="110" height="44" class="logits"/>
  <text x="378" y="120" class="label">o₁ / y₁</text>
  <rect x="520" y="92" width="110" height="44" class="logits"/>
  <text x="548" y="120" class="label">o₂ / y₂</text>
  <rect x="690" y="92" width="110" height="44" class="logits"/>
  <text x="718" y="120" class="label">o₃ / y₃</text>

  <rect x="400" y="320" width="90" height="44" class="box"/>
  <text x="432" y="348" class="label">x₂</text>
  <rect x="520" y="320" width="110" height="44" class="embed"/>
  <text x="558" y="348" class="label">e₂</text>
  <rect x="740" y="320" width="90" height="44" class="box"/>
  <text x="772" y="348" class="label">x₃</text>
  <rect x="850" y="320" width="110" height="44" class="embed"/>
  <text x="888" y="348" class="label">e₃</text>

  <path d="M150 312 L180 312" class="arrow"/>
  <path d="M290 312 C320 312 330 280 350 260" class="arrow"/>
  <path d="M450 230 L490 230" class="arrow"/>
  <path d="M620 230 L660 230" class="arrow"/>
  <path d="M450 220 L405 136" class="arrow"/>
  <path d="M620 220 L575 136" class="arrow"/>
  <path d="M790 220 L745 136" class="arrow"/>
  <path d="M490 342 C500 342 505 300 520 260" class="arrow"/>
  <path d="M830 342 C845 342 850 300 860 260" class="arrow"/>

  <text x="332" y="184" class="small">RNNCell(e₁, h₀)</text>
  <text x="502" y="184" class="small">RNNCell(e₂, h₁)</text>
  <text x="672" y="184" class="small">RNNCell(e₃, h₂)</text>
  <text x="296" y="154" class="small">W_xh, W_hh</text>
  <text x="467" y="154" class="small">W_xh, W_hh</text>
  <text x="638" y="154" class="small">W_xh, W_hh</text>
  <text x="385" y="84" class="small">W_hy</text>
  <text x="555" y="84" class="small">W_hy</text>
  <text x="725" y="84" class="small">W_hy</text>

  <path d="M720 74 C650 24 520 18 390 78" class="back"/>
  <text x="560" y="28" class="small">BPTT: gradients flow backward through time</text>
</svg>
```

- [ ] **Step 4: Update the week 6 section in `notes/transformer-study-path.md`**

```markdown
### 代码实践

- 对应仓库目录：`experiments/rnn-and-sequence-modeling/`
- 先看 `rnn_unrolled_computation_graph.svg`，把 `x_t -> e_t -> h_t -> logits` 和参数共享这条主线理顺
- 阅读 `char_rnn_next_char.py`，重点看 `CharRNNModel.forward_sequence()` 里 hidden state 如何按时间步更新
- 运行 `run_all.py`，观察 `outputs/char_rnn_training_curves.png`、`outputs/rnn_hidden_state_dynamics.png`、`outputs/rnn_prediction_trace.png` 和 `outputs/rnn_sequence_length_comparison.png`
- 最后阅读 `rnn_advantages_and_limits.md`，总结 RNN 为什么适合变长序列，以及为什么它在长依赖上会遇到困难
```

- [ ] **Step 5: Commit the docs and study-path update**

```bash
git add experiments/rnn-and-sequence-modeling/README.md \
        experiments/rnn-and-sequence-modeling/rnn_advantages_and_limits.md \
        experiments/rnn-and-sequence-modeling/rnn_unrolled_computation_graph.svg \
        notes/transformer-study-path.md
git commit -m "docs: add week 6 rnn learning materials"
```

### Task 6: Run End-to-End Verification

**Files:**
- Modify if needed: `experiments/rnn-and-sequence-modeling/char_rnn_next_char.py`
- Modify if needed: `experiments/rnn-and-sequence-modeling/rnn_sequence_length_probe.py`
- Modify if needed: `experiments/rnn-and-sequence-modeling/README.md`
- Modify if needed: `notes/transformer-study-path.md`

- [ ] **Step 1: Run the complete unit test suite for week 6**

Run: `python -m unittest discover -s experiments/rnn-and-sequence-modeling/tests -v`

Expected: PASS with all week 6 tests green.

- [ ] **Step 2: Run the week 6 unified entrypoint**

Run: `python experiments/rnn-and-sequence-modeling/run_all.py`

Expected: The script completes without exceptions and writes all five outputs into `experiments/rnn-and-sequence-modeling/outputs/`.

- [ ] **Step 3: Verify the output files exist**

Run: `rg --files experiments/rnn-and-sequence-modeling/outputs`

Expected:
- `experiments/rnn-and-sequence-modeling/outputs/char_rnn_training_curves.png`
- `experiments/rnn-and-sequence-modeling/outputs/rnn_hidden_state_dynamics.png`
- `experiments/rnn-and-sequence-modeling/outputs/rnn_prediction_trace.png`
- `experiments/rnn-and-sequence-modeling/outputs/rnn_sequence_length_comparison.png`
- `experiments/rnn-and-sequence-modeling/outputs/rnn_generated_samples.txt`

- [ ] **Step 4: Verify the docs point at the new experiment files**

Run: `rg -n 'experiments/rnn-and-sequence-modeling|rnn_unrolled_computation_graph|rnn_sequence_length_comparison' experiments/rnn-and-sequence-modeling/README.md notes/transformer-study-path.md`

Expected: Matches in both files, with week 6 pointing to the new directory and output filenames.

- [ ] **Step 5: Verify the Python files compile cleanly**

Run: `python -m compileall experiments/rnn-and-sequence-modeling`

Expected: No `SyntaxError`.

- [ ] **Step 6: Commit the verified implementation**

```bash
git add experiments/rnn-and-sequence-modeling notes/transformer-study-path.md
git commit -m "feat: complete week 6 rnn learning package"
```

## Self-Review Checklist

- Spec coverage:
  - Main char-level RNN implementation: Task 3
  - Sequence-length difficulty analysis: Task 4
  - README, concept note, and SVG: Task 5
  - Study-path alignment for the original user request: Task 5
  - Output verification and runtime checks: Task 6
- Placeholder scan:
  - No unfinished markers or “similar to above” shorthand remain
  - Every command names exact paths and expected outcomes
- Type consistency:
  - `CharRNNModel`, `build_demo_corpus`, `build_sequence_dataset`, `train_model`, `predict_next_chars`, and `run_probe` use consistent names across tasks and tests
