import sys; sys.path.insert(0,'.')
from data import CodeTokenizer, PythonCodeDataset
t = CodeTokenizer()

print(f"Tokenizer pad_token_id: {t.pad_token_id}")
print(f"Tokenizer bos_token_id: {t.bos_token_id}")
print(f"Tokenizer eos_token_id: {t.eos_token_id}")
print()

# Check from config
from config import CONFIG_SMALL
cfg = CONFIG_SMALL
print(f"Config pad_token_id: {cfg.pad_token_id}")
print()

# Test on a real code to see average lengths
sample = 'def hello():\n    print("hello")\n    return 42\n'
ids = t.encode(sample)
print(f"Sample '{sample[:20]}': {len(ids)} tokens (ids: {ids})")
print(f"  BOS at 0: id={ids[0]}")
print(f"  EOS at end: id={ids[-1]}")
print()

# Load a real sample from cache
cache = "D:/NeuroCoder/data_cache/train_cache.pt"
import torch
data = torch.load(cache, map_location="cpu", weights_only=False)
ex = data["examples"]
print(f"Cache has {len(ex)} samples")

# Check some random samples' lengths
import random
random.seed(42)
for i in random.sample(range(len(ex)), min(5, len(ex))):
    sample_ids = ex[i]
    real_len = len(sample_ids)
    # Count non-padding tokens
    non_pad = sum(1 for x in sample_ids if x != t.pad_token_id)
    pad_ratio = 1 - non_pad / cfg.max_seq_len
    print(f"  Sample {i}: total={real_len}, non-pad={non_pad}, pad_ratio={pad_ratio:.1%}")

print()
print("Diagnosis: if pad_ratio > 50%, the model's CE loss is mostly fake")
print("because pad tokens are ignored in loss (ignore_index=pad_token_id)")
