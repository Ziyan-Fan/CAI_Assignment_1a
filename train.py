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
    import os

    from scorer import span_scores

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.model == "crf":
        train_sentences = [example.tokens for example in train_data]
        train_labels = [example.slots for example in train_data]

        model.fit(train_sentences, train_labels)

        dev_sentences = [example.tokens for example in dev_data]
        dev_labels = [example.slots for example in dev_data]
        dev_predictions = model.predict(dev_sentences)

        dev_f1 = span_scores(dev_labels, dev_predictions)["f1"]

        import joblib

        joblib.dump(model.model, os.path.join(args.checkpoint_dir, "crf_model.pkl"))
        return float(dev_f1)

    if args.model == "bilstm":
        vocab = build_vocab([example.tokens for example in train_data])
        token_to_id = vocab["token_to_id"]
        unk_id = token_to_id.get("<UNK>", 0)

        tag_set = sorted({tag for example in train_data for tag in example.slots})
        tag_to_id = {tag: idx for idx, tag in enumerate(tag_set)}
        id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}

        model.vocab = type(
            "Vocab",
            (),
            {"token_to_id": token_to_id, "id_to_tag": id_to_tag},
        )()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        examples = []
        for example in train_data:
            token_ids = [token_to_id.get(tok, unk_id) for tok in example.tokens]
            label_ids = [tag_to_id[tag] for tag in example.slots]
            examples.append((token_ids, label_ids))

        best_dev_f1 = -1.0
        best_state = None

        for _ in range(10):
            model.train()

            for start in range(0, len(examples), 32):
                batch = examples[start : start + 32]
                max_len = max(len(tok_ids) for tok_ids, _ in batch)

                token_tensor = torch.zeros((len(batch), max_len), dtype=torch.long)
                label_tensor = torch.full((len(batch), max_len), -100, dtype=torch.long)
                lengths = []

                for i, (tok_ids, lab_ids) in enumerate(batch):
                    token_tensor[i, : len(tok_ids)] = torch.tensor(tok_ids, dtype=torch.long)
                    label_tensor[i, : len(lab_ids)] = torch.tensor(lab_ids, dtype=torch.long)
                    lengths.append(len(tok_ids))

                lengths_tensor = torch.tensor(lengths, dtype=torch.long)

                optimizer.zero_grad()
                logits = model(token_tensor, lengths_tensor)
                logits = logits.permute(0, 2, 1)
                loss = loss_fn(logits, label_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            dev_gold = []
            dev_pred = []

            with torch.no_grad():
                for example in dev_data:
                    ids = [token_to_id.get(tok, unk_id) for tok in example.tokens]
                    length = torch.tensor([len(ids)], dtype=torch.long)
                    padded = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

                    logits = model(padded, length)
                    pred_ids = logits[0, : length[0], :].argmax(dim=-1).tolist()
                    dev_pred.append([id_to_tag[idx] for idx in pred_ids])
                    dev_gold.append(example.slots)

            current_f1 = span_scores(dev_gold, dev_pred)["f1"]

            if current_f1 > best_dev_f1:
                best_dev_f1 = current_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint = {
            "state_dict": model.state_dict(),
            "vocab_size": vocab["vocab_size"],
            "tagset_size": len(tag_set),
            "vocab": {"token_to_id": token_to_id, "id_to_tag": id_to_tag},
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "bilstm_model.pt"))
        return float(best_dev_f1)

    raise ValueError(f"Unknown model type: {args.model!r}")


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
