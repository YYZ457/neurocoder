"""Quick check: analyze existing data cache"""
import sys; sys.path.insert(0,'.')
import torch

cache = torch.load('data_cache/train_cache.pt', weights_only=False)
examples = cache['examples']
n = len(examples)
print(f"Cached samples: {n}")

# Check padding on first 300 samples
pad_token_id = cache.get('pad_token_id', 0)
total_pos = 0
real_pos = 0
for i in range(min(300, n)):
    ex = examples[i]
    total_pos += len(ex)
    real = sum(1 for t in ex if t != pad_token_id)
    real_pos += real

print(f"First 300 samples: {real_pos}/{total_pos} real tokens = {real_pos/total_pos*100:.1f}%")
print(f"Effective tokens per sample: {real_pos/300:.0f}")
print(f"Effective tokens per step: {16 * real_pos/300:.0f}")
print(f"Tokens per parameter: {16 * real_pos/300 * 2541 / 193e6:.2f}")
