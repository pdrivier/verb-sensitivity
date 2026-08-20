"""
Score Start/End continuations of a passage with OLMo 2, using the FB-runner's
probability utilities (next_seq_prob / _answer_probabilities / MODELS) as the
scoring backbone, adapted to return log-probs and to fit the start/end jsonl
schema.

pip install -U torch transformers accelerate pandas numpy
"""

import ast
import json
import math
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model roster (from the FB runner). Uncomment more entries to compare models.
# ---------------------------------------------------------------------------

MODELS = {
    ### OLMo
    # "EleutherAI/pythia-14m": "Pythia 14m",
    # "EleutherAI/pythia-1b": "Pythia 1B",
    # "EleutherAI/pythia-6.9b": "Pythia 6.9B",
    # "EleutherAI/pythia-12b": "Pythia 12B",
    # "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    # "allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    "allenai/OLMo-2-0425-1B": "OLMO 2 1B",
}

INPUT_JSONL = "input.jsonl"
OUTPUT_PATH = "olmo2_start_end_logprobs.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Probability utilities (from the FB runner), adapted to return log-probs.
# ---------------------------------------------------------------------------

def next_seq_logprob(model, tokenizer, seen, unseen):
    """Autoregressive log-probability of `unseen` continuing `seen`.

    Same math as the FB runner's next_seq_prob (teacher-force the
    ground-truth continuation token in at each step) and returns the
    summed log-probability directly instead of exponentiating -- more
    numerically stable, and it's what we actually want here.

    Uses KV caching: the prompt (`seen`) is run through the model once,
    and each subsequent continuation token is scored by feeding only
    that single new token plus the cached past_key_values, instead of
    re-running the whole growing sequence from scratch. This gives
    identical results to the uncached version but is O(n) forward
    passes over single tokens instead of O(n) forward passes over the
    whole growing sequence each time -- much faster for multi-token
    continuations, and for models/frameworks that support it.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(seen, return_tensors="pt").to(device)
    unseen_ids = tokenizer.encode(unseen)

    log_probs = []
    with torch.no_grad():
        # Prime the cache with the prompt; logits[0, -1] predicts the first
        # continuation token.
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[0, -1]

        for unseen_id in unseen_ids:
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=0)
            log_probs.append(next_token_log_probs[unseen_id])

            # Feed only the new (ground-truth) token, reusing the cache,
            # to get the logits that predict the *next* continuation token.
            next_token_tensor = torch.tensor([[unseen_id]], device=device)
            outputs = model(next_token_tensor, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1]

    total_log_prob = torch.stack(log_probs).sum()
    return total_log_prob.item()


def next_seq_prob(model, tokenizer, seen, unseen):
    """Kept for parity with the original FB-runner API (raw probability)."""
    return math.exp(next_seq_logprob(model, tokenizer, seen, unseen))


def _answer_probabilities(
    model,
    tokenizer,
    passage_text: str,
    question_text: str,
    correct_answer: str,
    distractor_answer: str,
):
    """Original FB-runner utility: log-probabilities for correct vs
    distractor completions, where the prompt is "{passage} {question}".
    Kept as-is (just switched to next_seq_logprob) in case you reuse it
    for a schema where passage and question are separate fields.
    """
    passage_clean = passage_text.replace(" [MASK].", "").strip()
    prompt = f"{passage_clean} {question_text.strip()}".rstrip()

    correct_prefixed = f" {correct_answer.strip()}"
    distractor_prefixed = f" {distractor_answer.strip()}"

    logp_correct = next_seq_logprob(model, tokenizer, prompt, correct_prefixed)
    logp_distr = next_seq_logprob(model, tokenizer, prompt, distractor_prefixed)

    return logp_correct, logp_distr


def start_end_logprobs(model, tokenizer, passage: str, start_word: str, end_word: str):
    """Start/End variant for this jsonl's schema: `passage` already ends
    exactly where the completion belongs (no separate question to
    append -- `query` is just the tail of `passage`, not something to
    re-append). Leading-space + mask-cleaning conventions match
    `_answer_probabilities` above for consistency.
    """
    passage_clean = passage.replace(" [MASK].", "").rstrip()
    start_prefixed = f" {start_word.strip()}"
    end_prefixed = f" {end_word.strip()}"

    start_logprob = next_seq_logprob(model, tokenizer, passage_clean, start_prefixed)
    end_logprob = next_seq_logprob(model, tokenizer, passage_clean, end_prefixed)
    return start_logprob, end_logprob


# ---------------------------------------------------------------------------
# Checkpoint / revision sweeping utilities (from the FB runner).
# Not wired into main() below -- kept for reuse if you want per-checkpoint
# logprob trajectories later (see commented example at bottom of main()).
# ---------------------------------------------------------------------------

def sample_log_indices(k, mylist):
    """k: number of points to sample from list"""
    if k > len(mylist):
        raise ValueError("k cannot be larger than the length of the list")
    oversample_factor = 2
    raw = np.logspace(0, np.log10(len(mylist) - 1), num=k * oversample_factor)
    indices = np.unique(raw.astype(int))
    if indices[-1] != (len(mylist) - 1):
        indices = np.hstack((indices, len(mylist) - 1))
    if indices[0] != 0:
        indices = np.hstack((0, indices))
    while len(indices) < k:
        oversample_factor += 1
        raw = np.logspace(0, np.log10(len(mylist) - 1), num=k * oversample_factor)
        indices = np.unique(raw.astype(int))
        if indices[-1] != (len(mylist) - 1):
            indices = np.hstack((indices, len(mylist) - 1))
        if indices[0] != 0:
            indices = np.hstack((0, indices))
    return indices


def get_revision_list(model_path: str, all_revisions: list[str]) -> list[str]:
    """Return a revision list with stage-aware or fallback log sampling."""
    def parse_step(x):
        match = re.search(r"step(\d+)", x)
        return int(match.group(1)) if match else float("inf")

    checkpoints_sorted = sorted(all_revisions, key=parse_step)
    stage1_ckpts = [c for c in checkpoints_sorted if "stage1" in c]
    stage2_ckpts = [c for c in checkpoints_sorted if "stage2" in c]
    min_k_stage1 = 40
    if stage1_ckpts and stage2_ckpts:
        print(f"Found stage1 ({len(stage1_ckpts)}) and stage2 ({len(stage2_ckpts)}) checkpoints.")
        logstage1 = sample_log_indices(min_k_stage1, stage1_ckpts)
        selected1 = [stage1_ckpts[i] for i in logstage1]

        ingredients_list = [int(c.split("ingredient")[-1][0]) for c in stage2_ckpts]
        n_ingredients = np.unique(ingredients_list)
        min_k_stage2 = 5
        selected2 = []
        for ingredient in n_ingredients:
            current_list = [c for c in stage2_ckpts if "ingredient" + str(ingredient) in c]
            logstage2 = sample_log_indices(min_k_stage2, current_list)
            selected2.append([current_list[i] for i in logstage2])
        all_selected = selected1 + selected2
        return [item for sublist in all_selected for item in (sublist if isinstance(sublist, list) else [sublist])]

    print(f"No stage1/stage2 structure found for {model_path}. Using fallback.")
    indices = sample_log_indices(min(min_k_stage1, len(checkpoints_sorted)), checkpoints_sorted)
    return [checkpoints_sorted[i] for i in indices]


# ---------------------------------------------------------------------------
# jsonl loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Loads a jsonl file. Falls back to ast.literal_eval per line in case
    entries are Python-dict-repr style (single quotes) rather than strict JSON."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(ast.literal_eval(line))
    return records


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    records = load_jsonl(INPUT_JSONL)
    all_rows = []

    for model_path, model_label in MODELS.items():
        print(f"Loading {model_label} ({model_path})...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        model.eval()
        if DEVICE == "cpu":
            model.to(DEVICE)

        for rec in records:
            passage = rec["passage"]
            start_word = rec["start"]
            end_word = rec["end"]

            passage_token_ids = tokenizer(passage, add_special_tokens=True).input_ids
            passage_tokens = tokenizer.convert_ids_to_tokens(passage_token_ids)

            start_logprob, end_logprob = start_end_logprobs(
                model, tokenizer, passage, start_word, end_word
            )

            row = dict(rec)  # keep every original metadata field
            row.update(
                {
                    "model_path": model_path,
                    "model_label": model_label,
                    "passage_tokens": passage_tokens,
                    "passage_token_ids": passage_token_ids,
                    "start_logprob": start_logprob,
                    "end_logprob": end_logprob,
                    "logprob_diff_start_minus_end": start_logprob - end_logprob,
                }
            )
            all_rows.append(row)

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # --- optional: per-checkpoint sweep, if `model_path` has revisions ---
        # from huggingface_hub import list_repo_refs
        # all_revisions = [r.name for r in list_repo_refs(model_path).branches]
        # revisions = get_revision_list(model_path, all_revisions)
        # for revision in revisions:
        #     model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, ...)
        #     ... repeat the scoring loop above, tagging rows with `revision` ...

    df = pd.DataFrame(all_rows)
    df.to_pickle(OUTPUT_PATH)  # pickle preserves list-valued token columns
    df.drop(columns=["passage_tokens", "passage_token_ids"]).to_csv(
        OUTPUT_PATH.replace(".pkl", "_summary.csv"), index=False
    )
    print(f"Scored {len(df)} rows across {len(MODELS)} model(s). Saved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    main()
