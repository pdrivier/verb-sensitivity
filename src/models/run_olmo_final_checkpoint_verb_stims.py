"""Run FB reduced stimuli on final pre-training checkpoints only.

For Pythia: step143000 (final checkpoint).
For OLMo 2: last checkpoint from log-spaced stage1 sample.
"""
import argparse
import json
import re
import warnings


import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
from del_models import clear_huggingface_cache


# ---------------------------------------------------------------------------
# List of Models
# ---------------------------------------------------------------------------

MODELS = {
    "EleutherAI/pythia-14m": "Pythia 14m",
    "EleutherAI/pythia-1b": "Pythia 1B",
    "EleutherAI/pythia-6.9b": "Pythia 6.9B",
    "EleutherAI/pythia-12b": "Pythia 12B",
    #"allenai/OLMo-2-0325-32B": "OLMO 2 32B",
    #"allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    #"allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    #"allenai/OLMo-2-0425-1B": "OLMO 2 1B",
    "allenai/Olmo-3-1025-7B": "OLMO 3 7B",
    "allenai/Olmo-3-32B-Think": "OLMo 3 32B-Think"
}

PYTHIA_FINAL_REVISION = "step143000"

# ---------------------------------------------------------------------------
# Relevant Utilities
# ---------------------------------------------------------------------------


def sample_log_indices(k, mylist):
    """k: number of points to sample from list"""

    n = len(mylist)
    if k < 1:
        raise ValueError("k must be at least 1")
    if n == 0:
        raise ValueError("mylist is empty")
    if k >= n:
        return list(range(n))
    
    # This function originally began below, but I was running into issues where a stage2-ingredient
    # only had a single checkpoint, and was throwing errors here because the actual k was smaller 
    # than my threshold
    #if k > len(mylist):
    #    raise ValueError("k cannot be larger than the length of the list")
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
        if "ingredient" in stage2_ckpts:
	        ingredients_list = [int(c.split("ingredient")[-1][0]) for c in stage2_ckpts]
	        n_ingredients = np.unique(ingredients_list)
	        min_k_stage2 = 5
	        selected2 = []
	        for ingredient in n_ingredients:
	            current_list = [c for c in stage2_ckpts if "ingredient" + str(ingredient) in c]
	            logstage2 = sample_log_indices(min_k_stage2, current_list)
	            selected2.append([current_list[i] for i in logstage2])
        else:
        	# just keep all of them for now, until you have a better sense for what you'd 
        	# like to explore with stage 2 -- this especially applies to Olmo 3 models
        	selected2 = stage2_ckpts   
        all_selected = selected1 + selected2
        return [item for sublist in all_selected for item in (sublist if isinstance(sublist, list) else [sublist])]

    print(f"No stage1/stage2 structure found for {model_path}. Using fallback.")
    if len(checkpoints_sorted) < 40:
    	# just keep all of them
    	indices = range(0,len(checkpoints_sorted))
    else: 
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
    	warnings.warn(f"No stage1 checkpoints in revision list for {model_path}")
    	last = revision_list[-1]
    	print(f"  Last checkpoint from revision list for {model_path}: {last}")
    else: 
    	last = stage1_from_list[-1]
    	print(f"  Last stage1 checkpoint from revision list for {model_path}: {last}")
    
    return last

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

def return_tokenized_passage(tokenizer, seen):
    input_ids = tokenizer.encode(seen)
    tokenized_passage = tokenizer.convert_ids_to_tokens(input_ids)

    return tokenized_passage


def main(model_path, filepath, revision=None, suffix="final"):
    savepath = f"data/processed/verb-factivity-reduced-stims/"
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
    	dtype=torch.bfloat16,   # or dtype=... in newer transformers
    	low_cpu_mem_usage=True,
	)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    results = []
    ### Run model
    with open(filepath, "r", encoding="utf-8") as file:
    	for line_number, line in enumerate(file, start=1):
    		if not line.strip():
    			continue
    		try:
    			items = json.loads(line)   # a list of dicts
    		except json.JSONDecodeError:
    			print(f"Error parsing line {line_number}: {line.strip()}")
    			continue

    		for row in items:
    			try:
    				reduced_passage = row["passage"]
    				start_location = " " + row["start"]
    				end_location = " " + row["end"]

    				start_prob = next_seq_prob(model, tokenizer, reduced_passage, start_location)
    				end_prob = next_seq_prob(model, tokenizer, reduced_passage, end_location)

    				if start_prob == 0 or end_prob == 0:
    					continue

    				tokenized_passage = return_tokenized_passage(tokenizer, reduced_passage)

    				results.append({
    					'start_prob': start_prob,
    					'end_prob': end_prob,
    					'passage': reduced_passage,
    					'tokenized_passage': tokenized_passage,
    					'start': row['start'],
    					'end': row['end'],
    					'knowledge_state_format': row['knowledge_state_format'],
    					'knowledge_state': row['knowledge_state'],
    					'cues_available': row['cues_available'],
    					'first_mention': row['first_mention'],
    					'recent_mention': row['recent_mention'],
    					'log_odds': np.log2(start_prob / end_prob),
    					'verb': row['verb'],
    					'verb_type': row['verb_type'],
    					'relocation_verb_tense': row['relocation_verb_tense'],
    					'query': row['query'],
    					})
    			except KeyError as e:
    				print(f"Missing key {e} on line {line_number}")

    ### Create DataFRame
    df_results = pd.DataFrame(results)
    df_results['model_path'] = model_path
    df_results['model_shorthand'] = MODELS[model_path]

    if revision and not "Olmo-3" in model_path:
        parts = revision.split("-")  # e.g., ['stage2', 'ingredient4', 'step102500', 'tokens860B']
        stage = next((p for p in parts if p.startswith("stage")), None)
        ingredient = next((p for p in parts if p.startswith("ingredient")), None)
        step = next((p for p in parts if p.startswith("step")), None)
        tokens = next((p for p in parts if p.startswith("tokens")), None)

        df_results['stage'] = stage
        df_results['ingredient'] = ingredient
        df_results['step'] = int(step.replace("step", "")) if step else None
        df_results['tokens_seen'] = tokens.replace("tokens", "") if tokens else None
    
    elif revision and "Olmo-3" in model_path:
    	parts = revision.split("_") 
    	step = next((p for p in parts if p.startswith("step")), None)
    	tokens = next((p for p in parts if p.startswith("tokens")), None)

    	df_results['stage'] = None
    	df_results['ingredient'] = None
    	df_results['step'] = int(step.replace("step", "")) if step else None
    	df_results['tokens_seen'] = tokens.replace("tokens", "") if tokens else None

    else:
    	df_results['stage'] = None
    	df_results['ingredient'] = None
    	df_results['step'] = None
    	df_results['tokens_seen'] = None


    df_results.to_csv(os.path.join(savepath,filename), index=False)



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

    	filepath = "data/raw/verb-stims.jsonl"

    	for model_path in MODELS.keys():
            is_pythia = "pythia" in model_path.lower()

            if is_pythia:
                rev = PYTHIA_FINAL_REVISION
            else:
                rev = get_last_stage1_from_revision_list(model_path)

            suffix = rev.replace("/", "_")
            print(f"Running FB for: {model_path} @ {rev}")
            main(model_path, filepath = filepath, revision=rev, suffix=suffix)
            clear_huggingface_cache()