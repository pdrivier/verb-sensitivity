"""
prepare-stims.py

Takes in a .csv containing columns with passages and a series of queries that must be appended 
to form a full prompt to a language model, where the queries vary according to the factivity of the 
mental state verb they contain (or the lack of mental state verb altogether). 

Returns a .jsonl file with entries corresponding to the final prompts and all the associated category
labels for later comparisons.

"""

import json
import os
import pandas as pd

from tqdm import tqdm


filename = "verb-sensitivity-stims.csv"
directory = "data/raw"
filepath = os.path.join(directory,filename)

df = pd.read_csv(filepath)
query_cols = [col for col in df.columns if "critical" in col]

# Define the verb types 
verb_dict = {"knows": "factive", 
	"understands": "factive", 
	"recognizes": "factive", 
	"thinks": "nonfactive",
	"assumes": "nonfactive",
	"believes": "nonfactive",
	"suspects": "nonfactive",
	"neutral": "neutral"
}

results = []
count = 0
for i,row in tqdm(df.iterrows()):
	for col in query_cols:

		# grab the verb to generate the relevant verb label in the json entry
		verb = col.split("_")[-1]

		if verb not in verb_dict:
			raise KeyError(
				f"No verb_type mapping for {verb!r} (from column {col!r})"
				f"Known verbs: {sorted(verb_dict)}"
				)
		verb_type = verb_dict[verb]

		# grab components to the full prompt, stripping potential whitespaces at the end
		reduced_passage = row["reduced_passage"].rstrip()
		query = row[col].rstrip() 

		# fuse, inserting whitespace between components
		passage = reduced_passage + " " + query 

		results.append({
			"index": count,
			"passage": passage,
			"query": query,
			"verb": verb,
			"verb_type": verb_type,
			"start": row["start"],
			"end": row["end"],
			"first_mention": row["first_mention"],
			"recent_mention": row["recent_mention"],
			"cues_available": row["information"],                      # (Knowledge Cue, Knowledge Cue + State)
			"knowledge_state": row["knowledge_state"],                 # (True Belief, False Belief, Neutral)
			"knowledge_state_format": row["knowledge_state_format"],   # (Implicit, Explicit, None)
			"relocation_verb_tense": row["relocation_verb_tense"]	   # (past, present)
			})

		count += 1


resultpath = os.path.join(directory, "verb-stims.jsonl")
with open(resultpath, "w") as f:
	f.write(json.dumps(results) + "\n")
