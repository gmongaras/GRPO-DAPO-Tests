import torch
import os
import re
import json
from copy import deepcopy
from datasets import load_dataset
import bitsandbytes as bnb
os.environ["HF_Home"] = "cache"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
with open(".env") as f:
    token = f.read().strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
quantization_config = BitsAndBytesConfig(load_in_8bit=False)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=token, use_fast=True, cache_dir="cache")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=token, cache_dir="cache", torch_dtype=torch.bfloat16, device_map="cuda:0")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", load_in_4bit=True, token=token, cache_dir="cache", device_map="auto")


batch_size = 2
group_batch_size = 8
num_steps = 1000
max_length = 512
G = 16
min_batch_size = 4
eps_low = 0.2
eps_high = 0.28
lr = 1e-6
desired_length = 400
max_length = 512
warmup_steps = 20
temperature = 0.7
use_resampling = False
num_steps_save = 10
save_path = "models/"

# Sutup tokenizer
tokenizer.model_max_length = max_length
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load dataset
dataset = load_dataset("openai/gsm8k", "main", cache_dir="./cache")
dataset_train = dataset["train"]
dataset_test = dataset["test"]

# Convert data to torch
dataset_train.set_format(type="torch", columns=["question", "answer"])
dataset_test.set_format(type="torch", columns=["question", "answer"])

# PyTorch random sampler
random_sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=batch_size*group_batch_size*num_steps)

system_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You will be given a question. First reason about the question and answer. Then answer the question.
The final solution must be in json format:
{
    "solution": "put your solution here"
}
Your solution should only include the final answer in the format above.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: 
""".strip()

system_prompt_replace = """
You will be given a question. First reason about the question and answer. Then answer the question.
The final solution must be in json format:
{
    "solution": "put your solution here"
}
Your solution should only include the final answer in the format above
""".strip()

def collate_fn(batch):
    questions = tokenizer([system_prompt + " " + batch[i]["question"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for i in range(0, len(batch))], truncation=True, padding="longest", padding_side="left", return_tensors="pt", max_length=tokenizer.model_max_length)
    answers = tokenizer([batch[i]["answer"].split("#### ")[-1] for i in range(0, len(batch))], truncation=True, padding="longest", padding_side="right", return_tensors="pt", max_length=tokenizer.model_max_length)
    return {
        "question_ids": questions["input_ids"].repeat(group_batch_size, 1),
        "question_mask": questions["attention_mask"].repeat(group_batch_size, 1),
        "answer": [batch[i]["answer"].split("#### ")[-1] for i in range(0, len(batch))]*group_batch_size,
        "answer_ids": answers["input_ids"].repeat(group_batch_size, 1),
        "answer_mask": answers["attention_mask"].repeat(group_batch_size, 1),
    }


# PyTorch data loader
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, 
    sampler=random_sampler_train,
    batch_size=batch_size, 
    collate_fn=collate_fn,
    
    num_workers=10,
    prefetch_factor=10,
    persistent_workers=True,
)

search_string = """
{\s*"solution": .*\s*}
""".strip()

# For the normal reward:
#   Returns 1 if the answer is correct, -1 otherwise
# For the length reward:
#   L_max is like a "desired length"
#   L_cache is the max length we are considering
#   Returns 0 if generation length is under the desired length (under L_max - L_cache)
#           a small negative penalty if the length is between L_max - L_cache and L_max
#           -1 if generation length is way too long (over L_max)
# Also returns True if correct and False if not
@torch.no_grad()
def get_reward(string, answer, length, L_max, L_cache, length_penalty=True):
    if type(string) != str:
        string = tokenizer.decode(string)
    rew = 0
    correct = False

    # Remove system prompt
    string = string.replace(system_prompt_replace, "")

    # Get the normal reward
    try:
        sol = json.loads(re.findall(search_string, string)[-1])["solution"]
        if str(sol) == answer:
            rew += 1
            correct = True
        else:
            rew += -1 + 0.1
    except:
        rew += -1

    if length_penalty:
        # Get the length reward
        if length <= L_max - L_cache:
            rew += 0
        elif length < L_max:
            rew += ((L_max-L_cache)-length)/L_cache
        else:
            rew += -1

    return rew, correct
    
policy = model

assert G % group_batch_size == 0, f"Group size ({G}) must be divisible by group batch size ({group_batch_size})"

# Batch buffer. This allows us to resample batches until we get a large enough batch size
batch_buffer_rewards = []
batch_buffer_sequences = []
batch_buffer_masks = []
batch_buffer_loss_masks = []
batch_buffer_tokens = []

# Optimizer on the policy
optim = bnb.optim.Adam(policy.parameters(), lr=lr, optim_bits=8)
# optim = torch.optim.AdamW(policy.parameters(), lr=lr)

grad_scaler = torch.amp.GradScaler("cuda")

total_batch_group_size = batch_size*group_batch_size

# Update the old policy model with the current model
old_policy = deepcopy(policy.cpu())
policy = policy.cuda()
old_policy.eval()
policy.eval()


for step, batch in enumerate(data_loader_train):
    # We want to do all this without grad. We will do a trick to
    # do this with grad later.
    with torch.no_grad():
        # Sample G outputs from the policy for
        rewards_by_prompt = [[] for i in range(0, batch_size)]
        text_by_prompt = [[] for i in range(0, batch_size)]
        masks_by_prompt = [[] for i in range(0, batch_size)]
        loss_masks_by_prompt = [[] for i in range(0, batch_size)]
        numcorrect_by_prompt = [0 for i in range(0, batch_size)]
        tokens_by_prompt = [[] for i in range(0, batch_size)]
        outputs = []
        for i in range(0, G//group_batch_size):
            # Get output on batch
            batch_output = policy.generate(input_ids=batch["question_ids"].cuda(), attention_mask=batch["question_mask"].cuda(), max_length=max_length, do_sample=True, temperature=temperature)

            # Pad the output
            batch_output = tokenizer.pad({"input_ids": batch_output}, padding="max_length", padding_side="left", max_length=max_length)["input_ids"]

            # Decode each of the outputs
            text_outputs = [tokenizer.decode(batch_output[j], skip_special_tokens=True) for j in range(0, total_batch_group_size)]

            # Get masks (True to attend. False to not attend)
            masks = (
                (batch_output != tokenizer.pad_token_id) &
                (batch_output != 128001) &
                (batch_output != 128004) &
                (batch_output != 128009)
            )

            # Lengths of all outputs
            lengths = masks.int().sum(-1)

            # Loss masks will mask the prompt as well as the other padding tokens
            loss_masks = masks.clone()
            loss_masks[:, :batch["question_mask"].shape[1]] = False

            # Get all rewards - normal and length
            for j in range(0, total_batch_group_size):
                # This just takes the global batch size and converts
                # it to a batchsize within groups.
                batch_idx = j % batch_size

                # Get the reward
                rew, correct = get_reward(text_outputs[j], batch["answer"][j], lengths[j].item(), max_length, desired_length)
                rewards_by_prompt[batch_idx].append(rew)
                numcorrect_by_prompt[batch_idx] += correct

                # Add masks and text outputs
                text_by_prompt[batch_idx].append(text_outputs[j])
                masks_by_prompt[batch_idx].append(masks[j].cpu())
                loss_masks_by_prompt[batch_idx].append(loss_masks[j].cpu())
                tokens_by_prompt[batch_idx].append(batch_output[j].cpu())

        # Look for prompts where the reward is either all correct or all wrong
        for i in range(0, batch_size):
            # Skip if all wrong or all correct. Otherwise add to the buffer
            if step <= warmup_steps or not use_resampling or (numcorrect_by_prompt[i] > 0 and numcorrect_by_prompt[i] < G):
                batch_buffer_rewards.append(rewards_by_prompt[i])
                batch_buffer_sequences.append(text_by_prompt[i])
                batch_buffer_masks.append(masks_by_prompt[i])
                batch_buffer_loss_masks.append(loss_masks_by_prompt[i])
                batch_buffer_tokens.append(tokens_by_prompt[i])

    # Get more data if the batch isn't large enough
    if len(batch_buffer_rewards) < batch_size:
        continue

    # Put policy in train mode
    policy.train()

    # Put old policy on the GPU
    old_policy = old_policy.to(torch.device("cuda:0"))

    # Convert the tokens and masks into tensors (batch_size, group_size)
    batch_buffer_rewards = torch.tensor(batch_buffer_rewards, dtype=torch.float)
    # batch_buffer_masks = torch.stack(batch_buffer_masks)
    batch_buffer_masks = torch.stack([torch.stack(row) for row in batch_buffer_masks])
    batch_buffer_loss_masks = torch.stack([torch.stack(row) for row in batch_buffer_loss_masks])
    batch_buffer_tokens = torch.stack([torch.stack(row) for row in batch_buffer_tokens])

    # Compute advantages by group
    with torch.no_grad():
        advantages = (batch_buffer_rewards - batch_buffer_rewards.mean(-1, keepdim=True)) / (batch_buffer_rewards.std(-1, keepdim=True) + 1e-8)

    # Weight for the entire batch for all groups is the
    # total number of tokens
    loss_weight = 1/batch_buffer_loss_masks.sum()

    # Total reward for logging
    total_reward = 0

    # Iterate over all batches
    for batch_num in range(0, G):
        # Get the advantages, masks, and tokens for this batch
        adv = advantages[:, batch_num].cuda()
        masks = batch_buffer_masks[:, batch_num].cuda()
        loss_masks = batch_buffer_loss_masks[:, batch_num].cuda()
        tokens = batch_buffer_tokens[:, batch_num].cuda()

        # We can get the outputs of both the old model and current model by doing a forward pass on each
        # with the given prompt.
        # Note that the output is just the probability of the selected token from each distribution
        labels = tokens[:, 1:]
        with torch.no_grad():
            old_probs = old_policy(tokens, attention_masks=masks).logits.softmax(-1)[:, :-1]
            old_probs = torch.gather(old_probs, 2, labels.unsqueeze(2)).squeeze(2)
        new_probs = policy(tokens, attention_masks=masks).logits.softmax(-1)[:, :-1]
        new_probs = torch.gather(new_probs, 2, labels.unsqueeze(2)).squeeze(2)

        # Relative probs
        rel_probs = new_probs / old_probs

        # Clip the probabilities and multiply by the advantage
        adv = adv[:, None].to(rel_probs.device)
        rel_probs = torch.min(rel_probs*adv, rel_probs.clamp(min=1-eps_low, max=1+eps_high)*adv)

        # Mask loss
        rel_probs = rel_probs * loss_masks[:, :-1]

        # Sum up all values. Take the negative to maximize. Weight for averaging
        reward = -(rel_probs.sum() * loss_weight)

        # Backprop
        reward.backward()

        total_reward += reward.item()

        del reward, rel_probs, adv, new_probs, old_probs, masks, loss_masks, tokens

    per_correct = sum(numcorrect_by_prompt)/(len(numcorrect_by_prompt)*G)
    print(f"Step {step}, Reward: {-total_reward}, Percentage correct: {per_correct}")

    # Put old policy on the CPU
    old_policy = old_policy.cpu()
    # torch.cuda.empty_cache()

    # Step optimizer
    optim.step()
    optim.zero_grad()

    # Reset buffers
    batch_buffer_rewards = []
    batch_buffer_sequences = []
    batch_buffer_masks = []
    batch_buffer_loss_masks = []
    batch_buffer_tokens = []

    # Update the old policy model with the current model
    del old_policy
    old_policy = deepcopy(policy.cpu())
    policy = policy.cuda()
    old_policy.eval()
    policy.eval()

    if step % num_steps_save == 0:
        torch.cuda.empty_cache()
        policy = policy.cpu()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(policy, save_path + f"policy_{step}.pt")
        policy = policy.cuda()
        torch.cuda.empty_cache()
