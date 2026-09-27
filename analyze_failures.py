import os

from evaluate import load_model, load_eval_data


# ============================================================
# Helper functions
# ============================================================

def mismatches(gold, pred):
    """Count token-level prediction errors."""
    assert len(gold) == len(pred)
    return sum(g != p for g, p in zip(gold, pred))


def token_differences(sentence, gold, crf_pred, bilstm_pred):
    """
    Return readable lines only for tokens where at least
    one model makes an error.
    """
    lines = []

    for i, (token, g, c, b) in enumerate(
        zip(sentence, gold, crf_pred, bilstm_pred)
    ):
        if g != c or g != b:
            lines.append(
                f"  token {i:2d}: {token:18s} "
                f"gold={g:28s} "
                f"CRF={c:28s} "
                f"BiLSTM={b}"
            )

    return lines


def format_example(number, case_type, example):
    """
    Format one selected test-set failure for manual analysis.
    """
    test_index, sentence, gold, crf_pred, bilstm_pred = example

    assert len(sentence) == len(gold)
    assert len(gold) == len(crf_pred)
    assert len(gold) == len(bilstm_pred)

    crf_errors = mismatches(gold, crf_pred)
    bilstm_errors = mismatches(gold, bilstm_pred)

    output = []

    output.append("=" * 100)
    output.append(f"Example {number}")
    output.append(f"Test index: {test_index}")
    output.append(f"Case type: {case_type}")
    output.append("=" * 100)

    output.append(f"Sentence: {' '.join(sentence)}")
    output.append("")

    output.append(f"Gold:    {gold}")
    output.append(f"CRF:     {crf_pred}")
    output.append(f"BiLSTM:  {bilstm_pred}")
    output.append("")

    output.append(
        f"Number of wrong tokens: "
        f"CRF={crf_errors}, "
        f"BiLSTM={bilstm_errors}"
    )

    output.append("")
    output.append("Token-level differences:")

    diffs = token_differences(
        sentence,
        gold,
        crf_pred,
        bilstm_pred,
    )

    if diffs:
        output.extend(diffs)
    else:
        output.append("  No token-level differences found.")

    output.append("")
    output.append("Your diagnosis:")
    output.append("  ________________________________________________")
    output.append("  ________________________________________________")
    output.append("  ________________________________________________")
    output.append("")

    return "\n".join(output)


# ============================================================
# Final model checkpoints
# ============================================================

crf_dir = "checkpoints/final_crf_100"
bilstm_dir = "checkpoints/final_bilstm_100"

for folder in [crf_dir, bilstm_dir]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Missing checkpoint directory: {folder}"
        )


print("Using checkpoints:")
print(f"  CRF:     {crf_dir}")
print(f"  BiLSTM:  {bilstm_dir}")


# ============================================================
# Load final models and test set
# ============================================================

crf_model = load_model(
    "crf",
    crf_dir,
)

bilstm_model = load_model(
    "bilstm",
    bilstm_dir,
)

sentences, gold_labels = load_eval_data("test")

crf_vocab = getattr(
    crf_model,
    "vocab",
    None,
)

bilstm_vocab = getattr(
    bilstm_model,
    "vocab",
    None,
)

crf_preds = crf_model.predict(
    sentences,
    crf_vocab,
)

bilstm_preds = bilstm_model.predict(
    sentences,
    bilstm_vocab,
)

assert len(sentences) == len(gold_labels)
assert len(sentences) == len(crf_preds)
assert len(sentences) == len(bilstm_preds)


# ============================================================
# Categorize test-set failures
# ============================================================

cases = {
    "CRF wrong, BiLSTM correct": [],
    "BiLSTM wrong, CRF correct": [],
    "Both wrong, different ways": [],
}


for test_index, (
    sentence,
    gold,
    crf_pred,
    bilstm_pred,
) in enumerate(
    zip(
        sentences,
        gold_labels,
        crf_preds,
        bilstm_preds,
    )
):

    crf_bad = crf_pred != gold
    bilstm_bad = bilstm_pred != gold

    example = (
        test_index,
        sentence,
        gold,
        crf_pred,
        bilstm_pred,
    )

    # CRF fails but BiLSTM gets whole sentence correct
    if crf_bad and not bilstm_bad:
        cases[
            "CRF wrong, BiLSTM correct"
        ].append(example)

    # BiLSTM fails but CRF gets whole sentence correct
    elif bilstm_bad and not crf_bad:
        cases[
            "BiLSTM wrong, CRF correct"
        ].append(example)

    # Both fail, but they make different predictions
    elif (
        crf_bad
        and bilstm_bad
        and crf_pred != bilstm_pred
    ):
        cases[
            "Both wrong, different ways"
        ].append(example)


# ============================================================
# Sort examples
#
# For model-specific failures, put examples with more
# mistakes first.
#
# For both-wrong examples, use total number of errors.
# ============================================================

cases["CRF wrong, BiLSTM correct"].sort(
    key=lambda x: mismatches(x[2], x[3]),
    reverse=True,
)

cases["BiLSTM wrong, CRF correct"].sort(
    key=lambda x: mismatches(x[2], x[4]),
    reverse=True,
)

cases["Both wrong, different ways"].sort(
    key=lambda x: (
        mismatches(x[2], x[3])
        + mismatches(x[2], x[4])
    ),
    reverse=True,
)


# ============================================================
# Select 10 examples
#
# Target:
#   4 CRF-only failures
#   3 BiLSTM-only failures
#   3 both-wrong failures
# ============================================================

target_counts = {
    "CRF wrong, BiLSTM correct": 4,
    "BiLSTM wrong, CRF correct": 3,
    "Both wrong, different ways": 3,
}

selected = []


for case_type, target in target_counts.items():

    examples = cases[case_type]

    for example in examples[:target]:
        selected.append(
            (case_type, example)
        )


# ============================================================
# If fewer than 10 examples were available in the target
# categories, fill the remaining slots with unused failures.
# ============================================================

if len(selected) < 10:

    already_selected = {
        example[0]
        for _, example in selected
    }

    remaining = []

    for case_type, examples in cases.items():

        for example in examples:

            test_index = example[0]

            if test_index in already_selected:
                continue

            gold = example[2]
            crf_pred = example[3]
            bilstm_pred = example[4]

            severity = (
                mismatches(gold, crf_pred)
                + mismatches(gold, bilstm_pred)
            )

            remaining.append(
                (
                    severity,
                    case_type,
                    example,
                )
            )

    remaining.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for _, case_type, example in remaining:

        if len(selected) >= 10:
            break

        selected.append(
            (
                case_type,
                example,
            )
        )


# ============================================================
# Sanity check
# ============================================================

if len(selected) < 10:
    print(
        f"WARNING: Only {len(selected)} suitable "
        "failure examples were found."
    )


# ============================================================
# Write Part D candidate file
# ============================================================

output_file = "part_d_examples.txt"


with open(
    output_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "PART D — TEST-SET ERROR ANALYSIS CANDIDATES\n"
    )
    f.write("=" * 100 + "\n\n")

    f.write(
        f"CRF checkpoint: {crf_dir}\n"
    )
    f.write(
        f"BiLSTM checkpoint: {bilstm_dir}\n\n"
    )

    f.write(
        "These are automatically selected test-set "
        "failures from the final 100% models.\n"
    )

    f.write(
        "Use the token-level differences below to write "
        "your own diagnosis for each example.\n\n"
    )

    for number, (
        case_type,
        example,
    ) in enumerate(
        selected,
        start=1,
    ):

        f.write(
            format_example(
                number,
                case_type,
                example,
            )
        )

        f.write("\n\n")


# ============================================================
# Terminal summary
# ============================================================

print()
print(
    f"Saved {len(selected)} examples to "
    f"{output_file}"
)

print("\nAvailable failures:")

for key, examples in cases.items():
    print(
        f"  {key}: {len(examples)}"
    )

print("\nSelected:")

for key in cases:

    count = sum(
        1
        for case_type, _ in selected
        if case_type == key
    )

    print(
        f"  {key}: {count}"
    )

print("\nSelected test indices:")

for number, (
    case_type,
    example,
) in enumerate(
    selected,
    start=1,
):
    print(
        f"  Example {number}: "
        f"test index {example[0]} "
        f"({case_type})"
    )