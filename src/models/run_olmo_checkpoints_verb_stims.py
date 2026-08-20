"""
run_olmo_checkpoints_verb_stims.py

Obtains language model log probs over item Start and End locations for each of the passages 
from verb-stims.jsonl

For each jsonl record like:
    {
      "index": 1151,
      "passage": "... Hannah goes to get the saddle from the",
      "start": "stable",
      "end": "hut",
      ... other metadata ...
    }

NOTE: requires first tokenizing each passage according to the LM in question

"""

import ast
import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config / Model Loading
# ---------------------------------------------------------------------------

MODEL_NAME = "allenai/OLMo-2-1124-7B"  
INPUT_JSONL = "input.jsonl"
OUTPUT_PATH = "olmo2_logprobs.pkl"  # pandas pickle preserves list/dict columns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    if DEVICE == "cpu":
        model.to(DEVICE)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Load jsonl file
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Loads a jsonl file. Falls back to ast.literal_eval per line in case
    entries are Python-dict-repr style (single quotes) rather than strict JSON."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
	            try:
	                records.append(json.loads(line))
	            except json.JSONDecodeError:
	                records.append(ast.literal_eval(line))
    return records
				

# ---------------------------------------------------------------------------
# Tokenize and score completions, accounting for whitespaces
# ---------------------------------------------------------------------------

def score_continuation(model, tokenizer, prompt: str, continuation_word: str, device: str):
    """
    Returns (total_logprob, continuation_token_ids, continuation_token_strs,
    prompt_token_ids) for `continuation_word` completing `prompt`.

    Whitespace handling: if `prompt` does not already end in whitespace, we
    prepend a space to the continuation before tokenizing, so that e.g.
    "hut" is tokenized the way it would be mid-sentence (" hut"), not the
    way it would be at the very start of a document ("hut").

    Multi-token handling: we concatenate prompt_ids + continuation_ids and
    do a single forward pass, then sum the log-probs of each continuation
    token conditioned on everything before it (teacher forcing) -- the joint 
    log P(continuation | prompt) even if the continuation
    splits into multiple subword tokens.
    """
    needs_space = len(prompt) > 0 and not prompt[-1].isspace()
    continuation_text = (" " if needs_space else "") + continuation_word

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    cont_ids = tokenizer(continuation_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    if cont_ids.shape[1] == 0:
        raise ValueError(f"Empty tokenization for continuation: {continuation_text!r}")

    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits  # [1, seq_len, vocab]

    prompt_len = prompt_ids.shape[1]
    cont_len = cont_ids.shape[1]

    # logits at position i predict token i+1, so the logits that predict our
    # continuation tokens live at positions [prompt_len-1, prompt_len+cont_len-2]
    relevant_logits = logits[0, prompt_len - 1 : prompt_len + cont_len - 1, :].float()
    log_probs = torch.log_softmax(relevant_logits, dim=-1)
    token_log_probs = log_probs.gather(1, cont_ids[0].unsqueeze(1)).squeeze(1)  # [cont_len]

    total_log_prob = token_log_probs.sum().item()
    cont_token_ids = cont_ids[0].tolist()
    cont_token_strs = tokenizer.convert_ids_to_tokens(cont_token_ids)
    prompt_token_ids = prompt_ids[0].tolist()

    return total_log_prob, cont_token_ids, cont_token_strs, prompt_token_ids


# ---------------------------------------------------------------------------
# (e) Main loop: build the dataframe
# ---------------------------------------------------------------------------

def main():
    model, tokenizer = load_model_and_tokenizer()
    records = load_jsonl(INPUT_JSONL)

    rows = []
    for rec in records:
        passage = rec["passage"]
        start_word = rec["start"]
        end_word = rec["end"]

        passage_token_ids = tokenizer(passage, add_special_tokens=True).input_ids
        passage_tokens = tokenizer.convert_ids_to_tokens(passage_token_ids)

        start_logprob, start_tok_ids, start_toks, _ = score_continuation(
            model, tokenizer, passage, start_word, DEVICE
        )
        end_logprob, end_tok_ids, end_toks, _ = score_continuation(
            model, tokenizer, passage, end_word, DEVICE
        )

        row = dict(rec)  # keep every original metadata field (verb, cues_available, etc.)
        row.update(
            {
                "passage_tokens": passage_tokens,
                "passage_token_ids": passage_token_ids,
                "start_tokens": start_toks,
                "start_token_ids": start_tok_ids,
                "start_logprob": start_logprob,
                "end_tokens": end_toks,
                "end_token_ids": end_tok_ids,
                "end_logprob": end_logprob,
                "logprob_diff_start_minus_end": start_logprob - end_logprob,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_pickle(OUTPUT_PATH)  # pickle keeps list-valued columns intact
    # also handy for a quick look / sharing:
    df.drop(columns=["passage_tokens", "passage_token_ids", "start_tokens", "end_tokens",
                      "start_token_ids", "end_token_ids"]).to_csv(
        OUTPUT_PATH.replace(".pkl", "_summary.csv"), index=False
    )
    print(f"Scored {len(df)} rows. Saved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    main()