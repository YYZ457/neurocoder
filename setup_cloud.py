#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCoder Cloud Setup — download 1GB+ of REAL Chinese data from GitHub.
"""
import os, subprocess
from pathlib import Path

os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, dest_dir, branch=""):
    name = repo.split("/")[-1]
    dest = Path(dest_dir) / name
    if dest.exists():
        return
    url = f"https://github.com/{repo}.git"
    cmd = GIT + ["clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        print(f"  [OK] {name}")
    except:
        print(f"  [FAIL] {name}")

# =============================================================
# BLOCK 1: Chinese novels + literature (300MB+)
# =============================================================
print("\n" + "=" * 60)
print("BLOCK 1: Chinese novels and literature")
print("=" * 60)

LIT_REPOS = [
    "kourgeorge/novel-chinese",
    "chinese-poetry/chinese-poetry",
    "alex-ren/Chinese-Literature-Generator",
    "DreamMaker0117/Chinese_Corpus",
    "kavgan/Chinese-NLP-Corpus",
    "InsaneLife/ChineseNLPCorpus",
]

for repo in LIT_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# BLOCK 2: Chinese news + web text (500MB+)
# =============================================================
print("\n" + "=" * 60)
print("BLOCK 2: Chinese news and web text")
print("=" * 60)

NEWS_REPOS = [
    "CLUEbenchmark/CLUECorpus2020",
    "brightmart/nlp_chinese_corpus",
    "crownpku/ChineseNLP",
    "fighting41love/Chinese_NLP",
    "lingluo/Chinese-Corpus",
    "nghuyong/WeiboPublicStance",
    "liuhuanyong/ChineseNLPCorpus",
    "SophonPlus/ChineseNlpCorpus",
]

for repo in NEWS_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# BLOCK 3: Chinese conversations + dialogues (200MB+)
# =============================================================
print("\n" + "=" * 60)
print("BLOCK 3: Chinese conversations")
print("=" * 60)

CHAT_REPOS = [
    "coderchen01/Chinese_Chat_Corpus",
    "chatopera/Sentence-Corpus",
    "liuhuanyong/ChineseNLPCorpus",
    "Songrb/Chinese_Dialogue_Corpus",
    "coai/Chinese-Dialogue-Datasets",
    "thu-coai/CDial-GPT",
    "LitchiCheng/Chinese-Dialogue-Data",
]

for repo in CHAT_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# BLOCK 4: Chinese tech + education (200MB+)
# =============================================================
print("\n" + "=" * 60)
print("BLOCK 4: Chinese tech and education")
print("=" * 60)

TECH_REPOS = [
    "jackfrued/Python-100-Days",
    "d2l-ai/d2l-zh",
    "xitu/gold-miner",
    "xitu/tensorflow-docs",
    "MLEveryday/100-Days-Of-ML-Code",
    "scutan90/DeepLearning-500-questions",
    "unsonn/Deep-Learning-Interview-Book",
    "CyC2018/CS-Notes",
]

for repo in TECH_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# BLOCK 5: Chinese social + misc (100MB+)
# =============================================================
print("\n" + "=" * 60)
print("BLOCK 5: Chinese social and miscellaneous")
print("=" * 60)

MISC_REPOS = [
    "yapcn/idiom",
    "IUUVR/Chinese-Corpus",
    "Oye93/Chinese-NLP-Corpus",
    "LIUMIAOLL/Chinese-Corpus",
    "HIT-SCIR/Chinese-Corpus",
    "zhangyilang/ChineseCorpus",
]

for repo in MISC_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# BLOCK 6: Python code (small subset for balance)
# =============================================================
print("\n" + "=" * 60)
print("BLOCK 6: Python code")
print("=" * 60)

PY_DIR = DATA_DIR / "python"
PY_DIR.mkdir(exist_ok=True)

PY_REPOS = [
    "pallets/flask",
    "pallets/click",
    "psf/black",
    "pydantic/pydantic",
    "kennethreitz/requests",
    "Textualize/rich",
    "pytest-dev/pytest",
]

for repo in PY_REPOS:
    clone(repo, PY_DIR)

# =============================================================
# TOKENIZER: Retrain with Chinese text included
# =============================================================
print("\n" + "=" * 60)
print("TOKENIZER: Retraining with Chinese + code...")
print("=" * 60)

# Collect all text files for tokenizer training
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

train_files = []
for ext in ["*.txt", "*.py", "*.md"]:
    train_files.extend(DATA_DIR.rglob(ext))

# Sample up to 20000 files for tokenizer training (for speed)
import random
random.seed(42)
sample = random.sample(train_files, min(len(train_files), 20000))

print(f"  Training on {len(sample)} files...")

# Write consolidated corpus
corpus_path = "_train_corpus.txt"
with open(corpus_path, "w", encoding="utf-8", errors="ignore") as out:
    for f in sample:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if len(text) > 50:
                out.write(text[:10000])  # Truncate very long files
                out.write("\n")
        except:
            pass

print(f"  Corpus: {os.path.getsize(corpus_path)/1024/1024:.1f} MB")

# Train
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.normalizer = normalizers.NFKC()

special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
trainer = trainers.BpeTrainer(vocab_size=32768, special_tokens=special_tokens,
                              min_frequency=2, show_progress=True)
tokenizer.train([corpus_path], trainer)

os.makedirs("tokenizer_cache", exist_ok=True)
tokenizer.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(corpus_path)
print(f"  Tokenizer saved (vocab={tokenizer.get_vocab_size()})")

# =============================================================
# SUMMARY
# =============================================================
text_files = []
for ext in ["*.txt", "*.md", "*.py", "*.json", "*.jsonl", "*.csv", "*.tsv"]:
    text_files.extend(DATA_DIR.rglob(ext))
text_files = [f for f in text_files if f.is_file()]
total_bytes = sum(f.stat().st_size for f in text_files)
dirs = len([d for d in DATA_DIR.iterdir() if d.is_dir()])

print(f"""
{'='*60}
  DONE
{'='*60}
  Source repos:     {dirs}
  Text files:       {len(text_files):,}
  Total size:       {total_bytes/1024/1024/1024:.2f} GB
  Est. tokens:      {total_bytes//4:,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
