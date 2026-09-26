import json
from evaluate import load_model, load_eval_data

def write_predictions(filename, sentences, predictions):
    with open(filename, "w") as f:
        for i, (tokens, slots) in enumerate(zip(sentences, predictions)):
            assert len(tokens) == len(slots)
            f.write(json.dumps({
                "index": i,
                "tokens": tokens,
                "slots": slots
            }) + "\n")

# Load test set
sentences, _ = load_eval_data("test")

# Load final models from current directory
crf_model = load_model("crf", ".")
bilstm_model = load_model("bilstm", ".")

# Generate predictions
crf_preds = crf_model.predict(
    sentences,
    getattr(crf_model, "vocab", None)
)

bilstm_preds = bilstm_model.predict(
    sentences,
    getattr(bilstm_model, "vocab", None)
)

assert len(sentences) == 893
assert len(crf_preds) == 893
assert len(bilstm_preds) == 893

write_predictions(
    "crf_predictions.jsonl",
    sentences,
    crf_preds
)

write_predictions(
    "bilstm_predictions.jsonl",
    sentences,
    bilstm_preds
)

print("Created crf_predictions.jsonl")
print("Created bilstm_predictions.jsonl")
print("Test examples:", len(sentences))