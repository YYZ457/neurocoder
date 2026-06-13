#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCoder Cloud Setup — clone REAL Chinese data repos from GitHub.
Run: python setup_cloud.py && python train.py --config small --steps 200000 --data chinese_data
"""
import os, subprocess, sys
from pathlib import Path

os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, dest_dir, branch=""):
    name = repo.split("/")[-1]
    dest = Path(dest_dir) / name
    if dest.exists():
        print(f"  [Skip] {name}")
        return
    url = f"https://github.com/{repo}.git"
    cmd = GIT + ["clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        print(f"  [OK] {name}")
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:100] if e.stderr else ""
        print(f"  [FAIL] {name}: {err}")

def count_files(d):
    return len(list(Path(d).rglob("*.*")))

print("=" * 60)
print("  Downloading REAL Chinese data from GitHub")
print("  (all text: novels, news, chat, code, poetry...)")
print("=" * 60)

# =============================================================
# 1. LARGE Chinese text corpora
# =============================================================
print("\n[1/4] Chinese text corpora...")

# Chinese-novel-corpus — real novels, ~500MB text
clone("kourgeorge/novel-chinese", DATA_DIR)

# Chinese poetry — 40K+ poems
clone("chinese-poetry/chinese-poetry", DATA_DIR)

# Chinese Wikipedia text dump (smaller fork)
clone("brightmart/nlp_chinese_corpus", DATA_DIR)

# Chinese news corpus
clone("CLUEbenchmark/CLUECorpus2020", DATA_DIR)

# =============================================================
# 2. Chinese conversation + chat data
# =============================================================
print("\n[2/4] Chinese conversations...")

# Real chat corpora from Weibo, etc.
clone("coderchen01/Chinese_Chat_Corpus", DATA_DIR)
clone("liuhuanyong/ChineseNLPCorpus", DATA_DIR)

# Chinese sentences/QA corpus
clone("chatopera/Sentence-Corpus", DATA_DIR)

# Chinese idiom + proverb data
clone("yapcn/idiom", DATA_DIR)

# =============================================================
# 3. Chinese tech & educational content
# =============================================================
print("\n[3/4] Chinese tech content...")

# Python-100-days — Chinese Python tutorial
clone("jackfrued/Python-100-Days", DATA_DIR)

# Deep learning book in Chinese
clone("d2l-ai/d2l-zh", DATA_DIR)

# Chinese tech translations
clone("xitu/gold-miner", DATA_DIR)

# =============================================================
# 4. Python code (small curated subset)
# =============================================================
print("\n[4/4] Python code...")
py_dir = DATA_DIR / "python"
py_dir.mkdir(exist_ok=True)

for repo in [
    "pallets/flask",
    "pallets/click",
    "psf/black",
    "pydantic/pydantic",
    "kennethreitz/requests",
    "Textualize/rich",
]:
    clone(repo, py_dir)

# =============================================================
# Summary
# =============================================================
text_files = []
for ext in ["*.txt", "*.md", "*.py", "*.json", "*.jsonl", "*.csv"]:
    text_files.extend(DATA_DIR.rglob(ext))
total_bytes = sum(f.stat().st_size for f in text_files if f.is_file())

print(f"""
{'='*60}
  DONE — {total_bytes/1024/1024/1024:.2f} GB of real Chinese + code data
{'='*60}

  Total text files: {len(text_files):,}
  Data size:        {total_bytes/1024/1024:.0f} MB

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
