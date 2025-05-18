from datasets import load_dataset

# Load all helpfulness/harmless subsets (share the same schema)
dataset = load_dataset("Anthropic/hh-rlhf", cache_dir="./cache")