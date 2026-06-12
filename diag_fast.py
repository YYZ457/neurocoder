"""Quick data pipeline check"""
import sys; sys.path.insert(0,'.')
from data import CodeTokenizer, PythonCodeDataset

tokenizer = CodeTokenizer()

# Count samples WITHOUT caching
dataset = PythonCodeDataset(
    "sample_data/source", tokenizer,
    max_seq_len=2048, cache_path=None
)

n = len(dataset)
print(f"Total samples (2048-token chunks): {n}")
print(f"Effective batch size: 2 × 8 = 16")
print(f"Steps per epoch: {n // 16}")

# Check padding ratio on first 500 samples
import torch
real_tokens = 0
total = 0
for i in range(min(500, n)):
    ex = dataset.examples[i]
    real = sum(1 for t in ex if t != tokenizer.pad_token_id)
    real_tokens += real
    total += len(ex)

print(f"\nSample padding check (first 500 samples):")
print(f"  Real tokens: {real_tokens}/{total} = {real_tokens/total*100:.1f}%")
print(f"  Effective training tokens per step: {16 * (real_tokens/500):.0f}")
