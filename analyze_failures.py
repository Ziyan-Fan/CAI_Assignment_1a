import os

from evaluate import load_model, load_eval_data


def mismatches(gold, pred):
    assert len(gold) == len(pred)
    return sum(g != p for g, p in zip(gold, pred))


def token_differences(sentence, gold, crf_pred, bilstm_pred):
    """Return readable lines only for tokens where at least one model is wrong."""
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
    test_index, sentence, gold, crf_pred, bilstm_pred = example

    assert len(sentence) == len(gold)
    assert len(gold) == len(crf_pred)
    assert len(gold) == len(bilstm_pred)

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
        f"CRF={mismatches(gold, crf_pred)}, "
        f"BiLSTM={mismatches(gold, bilstm_pred)}"
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
    output.append("")

    return "\n".join(output)


# --------------------------------------------------
# Checkpoints
# --------------------------------------------------

crf_dir = "checkpoints/error_analysis_crf"
bilstm_dir = "checkpoints/error_analysis_bilstm"

for folder in [crf_dir, bilstm_dir]:
    if not os.path.exists(folder):
        print(f"Missing checkpoint directory: {folder}")
        raise SystemExit(1)


# --------------------------------------------------
# Load models and test data
# --------------------------------------------------

crf_model = load_model("crf", crf_dir)
bilstm_model = load_model("bilstm", bilstm_dir)

sentences, gold_labels = load_eval_data("test")

crf_vocab = getattr(crf_model, "vocab", None)
bilstm_vocab = getattr(bilstm_model, "vocab", None)

crf_preds = crf_model.predict(sentences, crf_vocab)
bilstm_preds = bilstm_model.predict(sentences, bilstm_vocab)

assert len(sentences) == len(gold_labels)
assert len(sentences) == len(crf_preds)
assert len(sentences) == len(bilstm_preds)


# --------------------------------------------------
# Categorize failures
# --------------------------------------------------

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

    if crf_bad and not bilstm_bad:
        cases["CRF wrong, BiLSTM correct"].append(example)

    elif bilstm_bad and not crf_bad:
        cases["BiLSTM wrong, CRF correct"].append(example)

    elif crf_bad and bilstm_bad and crf_pred != bilstm_pred:
        cases["Both wrong, different ways"].append(example)


# --------------------------------------------------
# Sort by largest error difference
# --------------------------------------------------

for key in cases:
    cases[key].sort(
        key=lambda x: max(
            mismatches(x[2], x[3]),
            mismatches(x[2], x[4]),
        ),
        reverse=True,
    )


# --------------------------------------------------
# Select 10 total examples
#
# Target:
#   4 CRF-only failures
#   3 BiLSTM-only failures
#   3 both-wrong failures
# --------------------------------------------------

target_counts = {
    "CRF wrong, BiLSTM correct": 4,
    "BiLSTM wrong, CRF correct": 3,
    "Both wrong, different ways": 3,
}

selected = []

for case_type, target in target_counts.items():
    for example in cases[case_type][:target]:
        selected.append((case_type, example))


# --------------------------------------------------
# If fewer than 10 were found, fill from remaining
# unused examples
# --------------------------------------------------

if len(selected) < 10:

    already_selected = {
        example[0]
        for _, example in selected
    }

    remaining = []

    for case_type, examples in cases.items():
        for example in examples:
            if example[0] not in already_selected:

                severity = max(
                    mismatches(example[2], example[3]),
                    mismatches(example[2], example[4]),
                )

                remaining.append(
                    (severity, case_type, example)
                )

    remaining.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for _, case_type, example in remaining:
        if len(selected) >= 10:
            break

        selected.append(
            (case_type, example)
        )


# --------------------------------------------------
# Write output file
# --------------------------------------------------

output_file = "part_d_examples.txt"

with open(output_file, "w") as f:

    f.write("PART D — TEST-SET ERROR ANALYSIS CANDIDATES\n")
    f.write("=" * 100 + "\n\n")

    f.write(
        "These are 10 automatically selected test-set failures.\n"
        "Use the token-level differences to write your own diagnosis "
        "for each example.\n\n"
    )

    for number, (case_type, example) in enumerate(
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


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(f"Saved {len(selected)} examples to: {output_file}")

print("\nAvailable failures:")
for key, examples in cases.items():
    print(f"  {key}: {len(examples)}")

print("\nSelected:")
for key in cases:
    count = sum(
        1
        for case_type, _ in selected
        if case_type == key
    )
    print(f"  {key}: {count}")