"""Diagnose: check data diversity and loss correctness"""
import sys; sys.path.insert(0, '.')
import torch
from data import CodeTokenizer, PythonCodeDataset
from config import CONFIG_4060, NeuroCoderConfig
from model import NeuroCoder
from pathlib import Path

# 1. Check data
print("=" * 60)
print("1. DATA DIVERSITY CHECK")
print("=" * 60)

tokenizer = CodeTokenizer()
dataset = PythonCodeDataset(
    "sample_data/source", tokenizer,
    max_seq_len=2048, cache_path=None
)

n_samples = len(dataset)
print(f"\nTotal samples (2048-token chunks): {n_samples}")

# Check for duplicates
sample_hashes = set()
dup_count = 0
for i in range(min(n_samples, 5000)):
    h = hash(tuple(dataset.examples[i][:100]))
    if h in sample_hashes:
        dup_count += 1
    sample_hashes.add(h)
print(f"Duplicates in first 5000 samples: {dup_count}")

# Check padding ratio
total_tokens = 0
total_real = 0
for i in range(min(n_samples, 1000)):
    ex = dataset.examples[i]
    total_tokens += len(ex)
    total_real += sum(1 for t in ex if t != tokenizer.pad_token_id)
print(f"Real token ratio: {total_real}/{total_tokens} = {total_real/total_tokens*100:.1f}%")
print(f"Effective training tokens: {total_real:,} (first 1000 samples)")

# 2. Check model loss on a random batch
print("\n" + "=" * 60)
print("2. LOSS SANITY CHECK")
print("=" * 60)

cfg = CONFIG_4060
model = NeuroCoder(cfg)
model = model.cuda().eval()

# Random tokens
rand_ids = torch.randint(0, cfg.vocab_size, (2, 512), device="cuda")
labels = rand_ids.clone()

with torch.no_grad():
    out = model(rand_ids, labels=labels)
    rand_loss = out["ce_loss"].item()
print(f"Random tokens CE loss (should be ~log(32768) ≈ 10.4): {rand_loss:.4f}")

# Constant token
const_ids = torch.full((2, 512), 100, device="cuda", dtype=torch.long)
labels_c = const_ids.clone()
with torch.no_grad():
    out2 = model(const_ids, labels=labels_c)
    const_loss = out2["ce_loss"].item()
print(f"Constant token CE loss: {const_loss:.4f} (should be high)")

# Single token repeated
one_token = torch.randint(0, 100, (2, 512), device="cuda")
labels_one = one_token.clone()
with torch.no_grad():
    out3 = model(one_token, labels=labels_one)
    copied_loss = out3["ce_loss"].item()
print(f"Random token CE (should be ~10.4): {copied_loss:.4f}")

# 3. Check if aux_loss dominates
print("\n" + "=" * 60)
print("3. AUX LOSS CHECK")
print("=" * 60)
model.train()
rand_ids_grad = torch.randint(0, cfg.vocab_size, (1, 512), device="cuda")
labels_grad = rand_ids_grad.clone()
out4 = model(rand_ids_grad, labels=labels_grad)
print(f"Total loss: {out4['loss'].item():.4f}")
print(f"CE loss: {out4['ce_loss'].item():.4f}")
print(f"Aux loss: {out4['aux_loss'].item():.4f}")
print(f"Aux/Total ratio: {out4['aux_loss'].item()/out4['loss'].item()*100:.1f}%")

print("\n[DONE]")
