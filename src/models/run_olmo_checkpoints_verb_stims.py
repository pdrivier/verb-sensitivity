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

import argparse

import pandas as pd
import numpy as np
import transformers
import torch
import os
import random
import re

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs

from del_models import clear_huggingface_cache

# ---------------------------------------------------------------------------
# List of Models
# ---------------------------------------------------------------------------

MODELS = {
    ### OLMo
    #"EleutherAI/pythia-14m": "Pythia 14m",
    #"EleutherAI/pythia-1b": "Pythia 1B",
    #"EleutherAI/pythia-6.9b": "Pythia 6.9B",
    #"EleutherAI/pythia-12b": "Pythia 12B",
    "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    #"allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    #"allenai/OLMo-2-0425-1B": "OLMO 2 1B"
}

#MODELS = {
    ### OLMo
  #  'allenai/OLMo-2-1124-7B': 'OLMo 2 7B',
  #  'allenai/OLMo-2-1124-7B-SFT': 'OLMo 2 7B SFT',
  #  'allenai/OLMo-2-1124-7B-DPO': 'OLMo 2 7B DPO',
   # 'allenai/OLMo-2-1124-7B-Instruct': 'OLMO 2 7B Instruct', 
   # 'allenai/OLMo-2-1124-13B': 'OLMO 2 13B',
#    'allenai/OLMo-2-1124-13B-SFT': 'OLMO 2 13B SFT', 
   # 'allenai/OLMo-2-1124-13B-DPO': 'OLMo 2 13B DPO', 
   # 'allenai/OLMo-2-1124-13B-Instruct': 'OLMO 2 13B Instruct',
  #  'allenai/OLMo-2-0325-32B': 'OLMO 2 32B',
   # 'allenai/OLMo-2-0325-32B-SFT': 'OLMO 2 32B SFT', 
   # 'allenai/OLMo-2-0325-32B-Instruct': 'OLMO 2 32B Instruct',
   # 'allenai/OLMo-2-0325-32B-DPO': 'OLMO 2 32B DPO',
   # 'allenai/OLMo-2-0425-1B': 'OLMO 2 1B',
   # 'allenai/OLMo-2-0425-1B-SFT': 'OLMO 2 1B SFT',
   # 'allenai/OLMo-2-0425-1B-DPO': 'OLMO 2 1B DPO',
   # 'allenai/OLMo-2-0425-1B-Instruct': 'OLMO 2 1B Instruct',
 

#}

# ---------------------------------------------------------------------------
# Relevant Utilities
# ---------------------------------------------------------------------------

def sample_log_indices(k, mylist):
    """k: number of points to sample from list"""
    if k > len(mylist):
        raise ValueError("k cannot be larger than the length of the list")
    # Generate more points than needed, to reduce chances of duplicates
    oversample_factor = 2
    raw = np.logspace(0, np.log10(len(mylist) - 1), num=k * oversample_factor)
    indices = np.unique(raw.astype(int))
    if indices[-1] != (len(mylist) - 1):
        indices = np.hstack((indices, len(mylist)-1))
    if indices[0] != 0:
        indices = np.hstack((0, indices))
    # Redo everything with a larger oversampling factor if you end up with fewer than 
    # your intended target checkpoints
    while len(indices) < k: 
        oversample_factor += 1
        raw = np.logspace(0, np.log10(len(mylist) - 1), num=k * oversample_factor)
        indices = np.unique(raw.astype(int))
        if indices[-1] != (len(mylist) - 1):
            indices = np.hstack((indices, len(mylist)-1))
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
        
        # treat stage2 differently
        ingredients_list = [int(c.split("ingredient")[-1][0]) for c in stage2_ckpts]
        n_ingredients = np.unique(ingredients_list)
        min_k_stage2 = 5
        selected2 = []
        for ingredient in n_ingredients: 
            # filter stage2 checkpoints for those trained on the current ingredient
            current_list = [c for c in stage2_ckpts if "ingredient" + str(ingredient) in c]
            # grab the same logspaced indices for each ingredient in stage2
            logstage2 = sample_log_indices(min_k_stage2, current_list)
            selected2.append([current_list[i] for i in logstage2])
        all_selected = selected1 + selected2 
        return [item for sublist in all_selected for item in (sublist if isinstance(sublist, list) else [sublist])]
        
    print(f"No stage1/stage2 structure found for {model_path}. Using fallback.")
    indices = sample_log_indices(min(min_k_stage1, len(checkpoints_sorted)), checkpoints_sorted)
    return [checkpoints_sorted[i] for i in indices]


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
    device = next(model.parameters()).device  # get model's actual device
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
        # Append next token to input
        next_token_tensor = torch.tensor([[unseen_id]], device=device)
        input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
    total_log_prob = sum(log_probs)
    total_prob = torch.exp(total_log_prob)
    return total_prob.item()

def return_tokenized_passage(tokenizer, seen):
    input_ids = tokenizer.encode(seen, return_tensors="pt")
    tokenized_passage = tokenizer.convert_ids_to_tokens(input_ids)

    return tokenized_passage


def main(model_path, filepath, revision = None, suffix=None):

    # Set up save path, filename, etc.
    savepath = f"data/processed/verb-factivity/"
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

    if "/" in model_path:
        filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"
    else:
        filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"

    print(filename)
    print(savepath)


    # Skip if already computed
    output_path = os.path.join(savepath,filename)
    if os.path.exists(output_path):
        print(f"  Skipping {revision} — already exists at {filename}")
        return
    

    ### Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto",
        # use_auth_token=True
    )
    #tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    results = []
    ### Run model
    with open(filepath, "r", encoding = "utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)

                ### Run model
                reduced_passage = row["passage"]
                start_location = " " + row["start"]
                end_location = " " + row["end"]

                start_prob = next_seq_prob(model, tokenizer, reduced_passage, start_location)
                end_prob = next_seq_prob(model, tokenizer, reduced_passage, end_location)

                if start_prob == 0 or end_prob == 0:
                    continue

                verb = row['verb']
                tokenized_passage = return_tokenized_passage(tokenizer, reduced_passage)


                results.append({
                'start_prob': start_prob,
                'end_prob': end_prob,
                'passage': reduced_passage,
                'tokenized_passage': tokenized_passage,
                'start': row['start'],
                'end': row['end'],
                'knowledge_state_format': row['knowledge_state_format'],
                'knowledge_cue': row['knowledge_cue'],
                'knowledge_state': row['knowledge_state'],
                'cues_available': row['cues_available'],
                'first_mention': row['first_mention'],
                'recent_mention': row['recent_mention'],
                'log_odds': np.log2(start_prob / end_prob),
                'verb': verb, # TODO: double-check these rows in the jsonl to make sure
                'verb_type': row['verb'],
                'relocation_verb_tense': row['relocation_verb_tense'],
                'query': row['query']
                })

                pbar.update(1)

            except json.JSONDecodeError:
                print(f"Error parsing line {line_number}: {line.strip()}")

    ### Create DataFRame
    df_results = pd.DataFrame(results)
    df_results['model_path'] = model_path
    df_results['model_shorthand'] = MODELS[model_path]

    if revision:
        parts = revision.split("-")  # e.g., ['stage2', 'ingredient4', 'step102500', 'tokens860B']
        stage = next((p for p in parts if p.startswith("stage")), None)
        ingredient = next((p for p in parts if p.startswith("ingredient")), None)
        step = next((p for p in parts if p.startswith("step")), None)
        tokens = next((p for p in parts if p.startswith("tokens")), None)

        df_results['stage'] = stage
        df_results['ingredient'] = ingredient
        df_results['step'] = int(step.replace("step", "")) if step else None
        df_results['tokens_seen'] = tokens.replace("tokens", "") if tokens else None
    else:
        df_results['stage'] = None
        df_results['ingredient'] = None
        df_results['step'] = None
        df_results['tokens_seen'] = None


    df_results.to_csv(os.path.join(savepath,filename), index=False)


				
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run verb sensitivity stims")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional HuggingFace model id (e.g., EleutherAI/pythia-14m). "
            "If not set, iterate default roster with skip rules."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print correctness summary (overall and by Q1-4) after running",
    )
    args = parser.parse_args()



    if args.model is not None:
        print("Running:", args.model)
        main(args.model, summary=args.summary)
    else:

        filepath = "data/raw/verb-stims.jsonl"

        # Only run select Olmo 13B checkpoints
        for model_path in MODELS.keys():
            
            # Grab the list of available revisions for this model
            refs = list_repo_refs(model_path)
        
            # Combine and deduplicate all tag/branch names
            all_revisions = [b.name for b in refs.branches] + [t.name for t in refs.tags]
            if not all_revisions:
                print(f"No usable checkpoints found for {model_path}, skipping.")
                continue
                    
            revision_list = get_revision_list(model_path, all_revisions)
            print(revision_list)

            # revision_list = ['step143000']
        
        
            for rev in revision_list:
                print(f"Running FB with checkpoint: {rev}")
        
                revision = rev # pass revision into main
                suffix = rev.replace("/", "_") # to tag output files uniquely
                
                # Set up save path, filename, etc.
                savepath = f"data/processed/fb_local_multi_verb/"
                if not os.path.exists(savepath): 
                    os.makedirs(savepath)
            
                if "/" in model_path:
                    filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"
                else:
                    filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"
            
                # Skip this checkpoint's analysis if you've already run it before
                print("Checking if we've already run this analysis...")
                if os.path.exists(os.path.join(savepath,filename)):
                    print("Already run this model for this checkpoint.")
                    continue
        
                main(model_path=model_path,
                    filepath = filepath,
                     revision=revision,  
                     suffix=suffix)  

                #clear_huggingface_cache()
                
                
            