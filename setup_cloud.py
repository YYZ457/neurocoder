#!/usr/bin/env python3
"""NeuroCoder Cloud — GBs of REAL Chinese data from open HuggingFace datasets."""
import os, subprocess, json, random, sys
from pathlib import Path
os.environ["GIT_TERMINAL_PROMPT"] = "0"
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("  Installing datasets library...")
print("=" * 60)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "huggingface_hub"], check=True)

from datasets import load_dataset
from huggingface_hub import login

def save_docs(ds, out_path, max_docs, name):
    """Save documents from a streaming dataset to a text file."""
    if out_path.exists():
        print(f"  [Skip] {out_path.name}")
        return True
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= max_docs:
                break
            text = example.get("text", "")
            if len(text) > 50:
                f.write(text.strip() + "\n\n")
                count += 1
                if count % 50000 == 0:
                    print(f"    {count} docs...")
            if i % 100 == 0 and i > 0 and count == 0:
                pass  # No valid docs yet
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"  [OK] {name}: {count:,} docs ({mb:.0f} MB)")
    return count > 0

print("=" * 60)
print("  Downloading REAL Chinese text from HuggingFace")
print("=" * 60)

# =============================================================
# 1. OpenCSG Fineweb-edu-chinese (420B tokens, no auth)
# =============================================================
print("\n[1/4] Fineweb-edu-chinese (420B token corpus)...")
fw_dir = DATA_DIR / "fineweb"
fw_dir.mkdir(exist_ok=True)

try:
    ds = load_dataset("opencsg/chinese-fineweb-edu", split="train", streaming=True)
    save_docs(ds, fw_dir / "fineweb_sample.txt", 200000, "Fineweb")
except Exception as e:
    print(f"  [FAIL] Fineweb: {str(e)[:80]}")

# =============================================================
# 2. CCI3.0-HQ (500GB high-quality Chinese, open)
# =============================================================
print("\n[2/4] CCI3.0-HQ (500GB high-quality Chinese)...")
cci_dir = DATA_DIR / "cci3"
cci_dir.mkdir(exist_ok=True)

try:
    ds = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True)
    save_docs(ds, cci_dir / "cci3_hq_sample.txt", 150000, "CCI3-HQ")
except Exception as e:
    print(f"  [FAIL] CCI3-HQ: {str(e)[:80]}")
    # Try with token=None for public access
    try:
        ds = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True, token=None)
        save_docs(ds, cci_dir / "cci3_hq_sample.txt", 150000, "CCI3-HQ")
    except Exception as e2:
        print(f"  [FAIL] CCI3-HQ (no token): {str(e2)[:80]}")

# =============================================================
# 3. MNBVC (open Chinese corpus)
# =============================================================
print("\n[3/4] MNBVC Chinese corpus...")
mnbvc_dir = DATA_DIR / "mnbvc"
mnbvc_dir.mkdir(exist_ok=True)

try:
    ds = load_dataset("liwu/MNBVC", split="train", streaming=True)
    save_docs(ds, mnbvc_dir / "mnbvc_sample.txt", 100000, "MNBVC")
except Exception as e:
    print(f"  [FAIL] MNBVC: {str(e)[:80]}")

# =============================================================
# 4. Generate Chinese conversations (backup)
# =============================================================
print("\n[4/4] Chinese conversations (backup)...")
chat_dir = DATA_DIR / "_chats"
chat_dir.mkdir(exist_ok=True)
chat_file = chat_dir / "chats.txt"

if not chat_file.exists():
    rnd = random.Random(42)
    pairs = [
        ("你好", "你好！很高兴见到你！"),
        ("早上好", "早上好！"),
        ("今天天气怎么样？", "今天天气不错！"),
        ("你吃饭了吗？", "吃了！你呢？"),
        ("在干嘛？", "在和你聊天呀！"),
        ("心情怎么样？", "很好！"),
        ("晚安", "晚安，好梦！"),
        ("谢谢", "不客气！"),
        ("Python是什么？", "Python是一种编程语言。"),
        ("什么是AI？", "人工智能的简称。"),
    ]
    with open(chat_file, "w", encoding="utf-8") as f:
        for q, a in pairs:
            for _ in range(50000):
                f.write(f"用户: {q}\n助手: {a}\n\n")
    print(f"  [OK] {chat_file.stat().st_size/1024/1024:.0f} MB generated")

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

txt_files = [f for f in DATA_DIR.rglob("*.txt") if f.is_file() and f.stat().st_size > 1000]
random.seed(42)
sample = random.sample(txt_files, min(len(txt_files), 5000))

corpus = "_tc.txt"
with open(corpus, "w", encoding="utf-8", errors="ignore") as out:
    for f in sample:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: out.write(t[:10000]+"\n")
        except: pass

tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.normalizer = normalizers.NFKC()
trainer = trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"],
    min_frequency=2, show_progress=True)
tok.train([corpus], trainer)
os.makedirs("tokenizer_cache", exist_ok=True)
tok.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(corpus)

total = sum(f.stat().st_size for f in txt_files)
print(f"""
{'='*60}
  DONE — {total/1024/1024:.0f} MB
{'='*60}
  Files: {len(txt_files):,}
  Est. tokens: {total//3:,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
