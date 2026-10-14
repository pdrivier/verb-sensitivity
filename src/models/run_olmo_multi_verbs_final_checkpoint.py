"""Run FB control task on final pre-training checkpoints only.

For Pythia: step143000 (final checkpoint).
For OLMo 2: last checkpoint from log-spaced stage1 sample.
"""
import argparse
import re
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
from del_models import clear_huggingface_cache

MODELS = {
     #"EleutherAI/pythia-14m": "Pythia 14m",
    # "EleutherAI/pythia-1b": "Pythia 1B",
    # "EleutherAI/pythia-6.9b": "Pythia 6.9B",
    # "EleutherAI/pythia-12b": "Pythia 12B",
    "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    #"allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    #"allenai/OLMo-2-0425-1B": "OLMO 2 1B",
}

PYTHIA_FINAL_REVISION = "step143000"

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


def get_last_stage1_from_revision_list(model_path: str) -> str:
    """Get the last stage1 checkpoint from the log-spaced revision list."""
    refs = list_repo_refs(model_path)
    all_revisions = [b.name for b in refs.branches] + [t.name for t in refs.tags]
    if not all_revisions:
        raise ValueError(f"No usable checkpoints found for {model_path}")

    revision_list = get_revision_list(model_path, all_revisions)

    # Filter to just stage1 checkpoints from the revision list
    stage1_from_list = [r for r in revision_list if "stage1" in r]
    if not stage1_from_list:
        raise ValueError(f"No stage1 checkpoints in revision list for {model_path}")

    last = stage1_from_list[-1]
    print(f"  Last stage1 checkpoint from revision list for {model_path}: {last}")
    return last


def next_seq_prob(model, tokenizer, seen, unseen):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(seen, return_tensors="pt").to(device)
    unseen_ids = tokenizer.encode(unseen)
    log_probs = []
    for unseen_id in unseen_ids:
        with torch.no_grad():
            logits = model(input_ids).logits
        next_token_logits = logits[0, -1]
        next_token_probs = torch.softmax(next_token_logits, dim=0)
        prob = next_token_probs[unseen_id]
        log_probs.append(torch.log(prob))
        next_token_tensor = torch.tensor([[unseen_id]], device=device)
        input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
    total_log_prob = sum(log_probs)
    total_prob = torch.exp(total_log_prob)
    return total_prob.item()


def main(model_path, revision=None, suffix="final"):
    savepath = f"data/processed/fb_local_multi_verb/"
    os.makedirs(savepath, exist_ok=True)

    model_name = model_path.split("/")[-1]
    filename = f"fb-{model_name}-{suffix}.csv"
    output_path = os.path.join(savepath, filename)

    if os.path.exists(output_path):
        print(f"  Skipping {model_path} (rev={revision}) — already exists at {output_path}")
        return

    print(f"  Loading {model_path} revision={revision}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

    df_fb = pd.read_csv("data/raw/fb_multi_verbs.csv")

    results = []
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row["passage"].replace(" [MASK].", "").strip()
            start_location = " " + row["start"]
            end_location = " " + row["end"]

            start_prob = next_seq_prob(model, tokenizer, passage, start_location)
            end_prob = next_seq_prob(model, tokenizer, passage, end_location)

            if start_prob == 0 or end_prob == 0:
                continue

            results.append({
                "start_prob": start_prob,
                "end_prob": end_prob,
                "passage": row["passage"],
                "start": row["start"],
                "end": row["end"],
                "knowledge_cue": row["knowledge_cue"],
                "first_mention": row["first_mention"],
                "recent_mention": row["recent_mention"],
                "log_odds": np.log2(start_prob / end_prob),
                "condition": row["condition"],
                'verb_type': row["verb_type"],
                'verb': row['verb']
            })
            pbar.update(1)

    df_results = pd.DataFrame(results)
    df_results["model_path"] = model_path
    df_results["model_shorthand"] = MODELS[model_path]

    if revision:
        parts = revision.split("-")
        stage = next((p for p in parts if p.startswith("stage")), None)
        ingredient = next((p for p in parts if p.startswith("ingredient")), None)
        step = next((p for p in parts if p.startswith("step")), None)
        tokens = next((p for p in parts if p.startswith("tokens")), None)

        df_results["stage"] = stage
        df_results["ingredient"] = ingredient
        df_results["step"] = int(step.replace("step", "")) if step else None
        df_results["tokens_seen"] = tokens.replace("tokens", "") if tokens else None
    else:
        df_results["stage"] = None
        df_results["ingredient"] = None
        df_results["step"] = None
        df_results["tokens_seen"] = None

    df_results.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FB control task on final pre-training checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional HuggingFace model id. If not set, iterate full roster.",
    )
    args = parser.parse_args()

    if args.model is not None:
        is_pythia = "pythia" in args.model.lower()
        if is_pythia:
            rev = PYTHIA_FINAL_REVISION
        else:
            rev = get_last_stage1_from_revision_list(args.model)
        suffix = rev.replace("/", "_")
        main(args.model, revision=rev, suffix=suffix)
    else:
        for model_path in MODELS.keys():
            is_pythia = "pythia" in model_path.lower()

            if is_pythia:
                rev = PYTHIA_FINAL_REVISION
            else:
                rev = get_last_stage1_from_revision_list(model_path)

            suffix = rev.replace("/", "_")
            print(f"Running FB for: {model_path} @ {rev}")
            main(model_path, revision=rev, suffix=suffix)
            #clear_huggingface_cache()