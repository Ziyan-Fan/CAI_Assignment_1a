"""Course-provided skeleton for training Part A (CRF) and Part B (BiLSTM) models.

In this training script, CLI parsing, data loading, and model construction are wired up for you.
The only one thing left to implement is `train_loop()` — actual training loop,
for both Part A CRF and Part B BiLSTM. Look for the `# TODO: implement` block below.

Usage:
    python train.py --model crf    --train-fraction 1.00 --checkpoint-dir checkpoints/crf_1.00
    python train.py --model bilstm --train-fraction 0.25 --checkpoint-dir checkpoints/bilstm_0.25 \
        --tensorboard-logdir runs/bilstm_0.25
"""

from __future__ import annotations
import argparse
import random
from collections import Counter
from typing import Any
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Train a CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model to train.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training set to use, e.g. 0.05 / 0.10 / 0.25 / 1.00.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save the trained model to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default=None,
        help="TensorBoard log directory. Only used when --model BiLSTM; "
        "ignored for --model crf (CRF training doesn't need TensorBoard or a GPU).",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch — CPU + CUDA — for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(fraction: float, seed: int) -> tuple[Any, Any, Any, Any]:
    """Load the ATIS train/dev/test splits, subsampling the train split.

    Returns:
        (`train_data`, `dev_data`, `test_data`, `vocab`)

        `train_data`/`dev_data`/`test_data` are `list[data_loader.Example]`.
        `vocab` is the dict returned by `build_vocab()` (see below),
        augmented with `tagset_size` (via `data_loader.slot_label_set()`) —
        needed to construct `BiLSTMModel` in `build_model()`.
    """
    import data_loader

    all_splits = data_loader.load_atis()
    train_data = data_loader.subsample(
        all_splits["train"], fraction=fraction, seed=seed
    )

    dev_data = all_splits["dev"]
    test_data = all_splits["test"]

    train_sentences = [example.tokens for example in train_data]
    vocab = build_vocab(train_sentences)
    vocab["tagset_size"] = len(data_loader.slot_label_set(train_data))

    return train_data, dev_data, test_data, vocab


# Built here rather than in data_loader.py to avoid touching a
# teammate-owned file; if data_loader.py later grows its own vocab
# utility, this should be removed in favor of that to avoid two
# divergent implementations.
def build_vocab(train_sentences: list[list[str]]) -> dict[str, Any]:
    """Build a token vocabulary from tokenized training sentences.

    Fully implemented — this is infrastructure, not the student-fill part
    of the assignment.

    Reserves index 0 for "<PAD>" and index 1 for "<UNK>". Tokens are then
    ordered by descending frequency, ties broken alphabetically. This
    determinism is required: `token_to_id` must be identical across
    machines/Python versions/runs, or it risks breaking the ±0.3 F1
    reproducibility tolerance in the grading rubric. Do NOT rely on dict or
    set iteration order for this — the explicit `sorted()` key below is
    what makes it deterministic.

    Args:
        train_sentences: Tokenized training sentences.

    Returns:
        A dict with `token_to_id` (Dict[str, int]), `id_to_token`
        (Dict[int, str]), and `vocab_size` (int).
    """
    counts = Counter(token for sentence in train_sentences for token in sentence)
    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token))

    token_to_id = {"<PAD>": 0, "<UNK>": 1}
    for token in ordered_tokens:
        token_to_id[token] = len(token_to_id)
    id_to_token = {index: token for token, index in token_to_id.items()}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(token_to_id),
    }


def build_model(
    model_type: str,
    vocab_size: int | None = None,
    tagset_size: int | None = None,
) -> Any:
    """Construct an untrained model instance for the given model type.

    `vocab_size`/`tagset_size` (from `load_data()`'s `vocab`) are required
    for `model_type == "bilstm"` and ignored for `model_type == "crf"` —
    `CRFModel` works on raw token features, not a fixed vocabulary, so
    there's nothing for it to size itself against. This is intentional,
    not a bug, same as `CRFModel.predict()`'s unused `vocab` parameter.
    """
    if model_type == "crf":
        from models.crf import CRFModel

        return CRFModel()

    if model_type == "bilstm":
        from models.bilstm import BiLSTMModel

        if vocab_size is None or tagset_size is None:
            raise ValueError(
                "vocab_size and tagset_size are required for model_type='bilstm'"
            )

        return BiLSTMModel(vocab_size=vocab_size, tagset_size=tagset_size)

    raise ValueError(f"Unknown model type: {model_type!r}")


def train_loop(
    model: Any,
    train_data: Any,
    dev_data: Any,
    args: argparse.Namespace,
) -> float:
    """Train `model` on `train_data`, validating against `dev_data`.

    This is the part you implement.

    Your implementation must:
      - Train `model` on `train_data`.

      - Use `dev_data` for validation (and early stopping, if you choose to implement it).

      - When `args.model == "BiLSTM"`, log training progress to
        TensorBoard via `args.tensorboard_logdir` (e.g. using
        `torch.utils.tensorboard.SummaryWriter`). Not applicable for `args.model == "crf"`.

      - Save the final trained model to `args.checkpoint_dir`. See
        `evaluate.py`'s `load_model()` docstring for the exact checkpoint
        format your saved model must match.

    Returns:
        Final dev-set slot F1 (float). `main()` will print what you return.
    """
    from pathlib import Path
    from types import SimpleNamespace
    import os
    from evaluate import compute_metrics

    if not train_data or not dev_data:
        raise ValueError("Training and development sets must be nonempty.")
    for example in list(train_data) + list(dev_data):
        if not example.tokens or len(example.tokens) != len(example.slots):
            raise ValueError("Each example must contain aligned, nonempty tokens and slots.")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_sentences = [example.tokens for example in train_data]
    train_labels = [example.slots for example in train_data]
    dev_sentences = [example.tokens for example in dev_data]
    dev_labels = [example.slots for example in dev_data]

    if args.model == "crf":
        import joblib

        model.fit(train_sentences, train_labels)
        score = compute_metrics(model.predict(dev_sentences), dev_labels)["span_f1"]
        joblib.dump(model.model, checkpoint_dir / "crf_model.pkl")
        print(f"CRF features: word/case/digits/affixes/context +/-2/bigrams; dev F1={score:.4f}")
        return score
    if args.model != "bilstm":
        raise ValueError(f"Unknown model type: {args.model!r}")

    vocab = build_vocab(train_sentences)
    tags = sorted({tag for labels in train_labels for tag in labels})
    vocab["tag_to_id"] = {tag: index for index, tag in enumerate(tags)}
    vocab["id_to_tag"] = dict(enumerate(tags))
    token_to_id, tag_to_id = vocab["token_to_id"], vocab["tag_to_id"]
    # Default to an available NVIDIA GPU; allow CPU runs on GPU machines
    # without changing the starter's command-line interface.
    requested_device = os.environ.get("ATIS_DEVICE", "auto").strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("ATIS_DEVICE must be 'auto', 'cpu', or 'cuda'.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ATIS_DEVICE=cuda requires an available CUDA GPU and CUDA-enabled PyTorch.")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model.to(device)
    print(f"Training device: {device}")

    # Keep the bonus CRF parameters outside the module because the supplied
    # loader constructs the unchanged BiLSTM before restoring its state_dict.
    # They are optimized jointly, then serialized as plain vocabulary lists.
    transitions = torch.nn.Parameter(torch.zeros(len(tags), len(tags), device=device))
    start = torch.nn.Parameter(torch.zeros(len(tags), device=device))
    end = torch.nn.Parameter(torch.zeros(len(tags), device=device))
    parameters = list(model.parameters()) + [transitions, start, end]
    optimizer = torch.optim.Adam(parameters, lr=0.003, weight_decay=1e-5)
    encoded = [
        (
            torch.tensor([token_to_id.get(token, 1) for token in sentence]),
            torch.tensor([tag_to_id[tag] for tag in labels]),
        )
        for sentence, labels in zip(train_sentences, train_labels)
    ]

    model.vocab = SimpleNamespace(
        token_to_id=token_to_id,
        id_to_tag=vocab["id_to_tag"],
    )

    best_dev_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    max_epochs = 15
    patience = 3

    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(encoded))

        for start in range(0, len(encoded), 32):
            batch_indices = permutation[start : start + 32]
            batch = [encoded[idx] for idx in batch_indices]
            max_len = max(len(tokens) for tokens, _ in batch)

            token_tensor = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=device
            )
            label_tensor = torch.full(
                (len(batch), max_len), -100, dtype=torch.long, device=device
            )
            lengths = []

            for i, (token_ids, label_ids) in enumerate(batch):
                token_tensor[i, : len(token_ids)] = token_ids.to(device)
                label_tensor[i, : len(label_ids)] = label_ids.to(device)
                lengths.append(len(token_ids))

            lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(token_tensor, lengths_tensor)
            logits = logits.permute(0, 2, 1)
            loss = torch.nn.functional.cross_entropy(
                logits, label_tensor, ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()

        model.eval()
        dev_gold = []
        dev_pred = []
        with torch.no_grad():
            for example in dev_data:
                token_ids = [token_to_id.get(token, 1) for token in example.tokens]
                lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
                padded = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(padded, lengths)
                pred_ids = logits[0, : lengths[0], :].argmax(dim=-1).cpu().tolist()
                dev_pred.append([vocab["id_to_tag"][idx] for idx in pred_ids])
                dev_gold.append(example.slots)

        current_f1 = compute_metrics(dev_pred, dev_gold)["span_f1"]

        if current_f1 > best_dev_f1:
            best_dev_f1 = current_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab_size": vocab["vocab_size"],
        "tagset_size": len(tags),
        "vocab": {"token_to_id": token_to_id, "id_to_tag": vocab["id_to_tag"]},
    }
    torch.save(checkpoint, checkpoint_dir / "bilstm_model.pt")
    return float(best_dev_f1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_data, dev_data, test_data, vocab = load_data(args.train_fraction, args.seed)

    if args.model == "bilstm":
        model = build_model(
            args.model, vocab_size=vocab["vocab_size"], tagset_size=vocab["tagset_size"]
        )

    else:
        model = build_model(args.model)

    dev_score = train_loop(model, train_data, dev_data, args)

    import os

    if not os.path.isdir(args.checkpoint_dir) or not os.listdir(args.checkpoint_dir):
        raise RuntimeError(
            f"Expected train_loop() to save a checkpoint to {args.checkpoint_dir!r}, "
            "but the directory is missing or empty."
        )

    print("Training complete.")
    print(f"  Model:          {args.model}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    if dev_score is not None:
        print(f"  Final dev F1:   {dev_score:.4f}")


if __name__ == "__main__":
    main()
