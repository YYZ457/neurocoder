#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCoder Cloud Setup — downloads REAL Chinese + code data from GitHub.
Run: python setup_cloud.py && python train.py --config small --steps 200000 --data chinese_data

Downloads real Chinese text (Wikipedia, news, conversations) and Python repos.
All converted to .txt format compatible with the data pipeline.
"""
import os, subprocess, urllib.request, json, random, re
from pathlib import Path

DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def dl(url, dest):
    if dest.exists():
        print(f"  [Skip] {dest.name}")
        return
    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)

def clone(repo, dest_dir, branch=""):
    name = repo.split("/")[-1]
    dest = Path(dest_dir) / name
    if dest.exists():
        return
    url = f"https://github.com/{repo}.git"
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([url, str(dest)])
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=True)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}")

print("=" * 60)
print("  Downloading REAL Chinese data from GitHub & web")
print("=" * 60)

# =============================================================
# 1. Chinese Wikipedia (plain text, preprocessed) — ~400MB
# =============================================================
print("\n[1/5] Chinese Wikipedia text...")
wiki_dir = DATA_DIR / "wiki"
wiki_dir.mkdir(exist_ok=True)
# Use a pre-processed Wikipedia dump (plain text, one line per article)
dl("https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/zhwiki/zhwiki_articles.txt",
   wiki_dir / "zhwiki.txt")
# Alternative: download from HuggingFace mirror
dl("https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.jsonl",
   wiki_dir / "wikipedia_cn.jsonl")

# =============================================================
# 2. Chinese news corpus — CLUECorpus
# =============================================================
print("\n[2/5] Chinese news data...")
news_dir = DATA_DIR / "news"
news_dir.mkdir(exist_ok=True)
dl("https://raw.githubusercontent.com/CLUEbenchmark/CLUECorpus2020/main/raw_data/news_sohusite_500k.jsonl",
   news_dir / "news_sohu.jsonl")

# =============================================================
# 3. Chinese conversation/chat data from GitHub
# =============================================================
print("\n[3/5] Chinese conversation datasets...")
chat_dir = DATA_DIR / "chats"
chat_dir.mkdir(exist_ok=True)

# Download BELLE-style instruction data (small, manageable)
belle_urls = [
    ("https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/chatgpt_instruct/chatgpt_instruct_10k.json",
     "belle_10k.json"),
]
for url, name in belle_urls:
    dl(url, chat_dir / name)

# Clone Chinese conversation repos (small, focused)
for repo in ["coderchen01/Chinese_Chat_Corpus", "chatopera/Sentence-Corpus"]:
    clone(repo, chat_dir)

# Download Chinese subtitles (natural conversation)
dl("https://raw.githubusercontent.com/liuhuanyong/ChineseNLPCorpus/master/data/weibo/weibo_all.txt",
   chat_dir / "weibo.txt")

# =============================================================
# 4. Chinese literature + culture
# =============================================================
print("\n[4/5] Chinese literature and text...")
lit_dir = DATA_DIR / "literature"
lit_dir.mkdir(exist_ok=True)

# Chinese poetry
clone("chinese-poetry/chinese-poetry", lit_dir)

# Download some Chinese novels (preprocessed)
novels = [
    ("https://raw.githubusercontent.com/kourgeorge/novel-chinese/master/novels/三国演义.txt", "三国演义.txt"),
]
for url, name in novels:
    dl(url, lit_dir / name)

# =============================================================
# 5. Curated Python code
# =============================================================
print("\n[5/5] Python code (curated subset)...")
py_dir = DATA_DIR / "python"
py_dir.mkdir(exist_ok=True)

for repo, name in [
    ("pallets/flask", "flask"),
    ("pydantic/pydantic", "pydantic"),
    ("psf/black", "black"),
    ("Textualize/rich", "rich"),
    ("kennethreitz/requests", "requests"),
    ("pallets/click", "click"),
]:
    clone(repo, py_dir)

# =============================================================
# Convert everything to .txt for the data pipeline
# =============================================================
print("\nConverting to .txt format...")

CONVERTED_DIR = DATA_DIR / "_converted"
CONVERTED_DIR.mkdir(exist_ok=True)

# Convert JSONL chat data (extract text fields)
for f in DATA_DIR.rglob("*.jsonl"):
    out_name = f"chat_{f.parent.name}_{f.stem}.txt"
    out_path = CONVERTED_DIR / out_name
    if out_path.exists():
        continue
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for line in f.open(encoding="utf-8", errors="ignore"):
            try:
                item = json.loads(line)
                text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                if len(text) > 20:
                    out.write(text + "\n")
                    count += 1
            except:
                pass
    print(f"  [OK] {out_name} ({count} lines)")

# Convert JSON files (list format)
for f in DATA_DIR.rglob("*.json"):
    out_name = f"json_{f.parent.name}_{f.stem}.txt"
    out_path = CONVERTED_DIR / out_name
    if out_path.exists() or f.name == "package.json":
        continue
    try:
        data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, list):
            with open(out_path, "w", encoding="utf-8") as out:
                for item in data:
                    if isinstance(item, str):
                        out.write(item + "\n")
                    elif isinstance(item, dict):
                        text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                        if len(text) > 20:
                            out.write(text + "\n")
            print(f"  [OK] {out_name}")
    except:
        pass

# =============================================================
# Summary
# =============================================================
all_files = list(DATA_DIR.rglob("*.*"))
total_bytes = sum(f.stat().st_size for f in all_files if f.is_file())
txt_files = list(DATA_DIR.rglob("*.txt"))
md_files = list(DATA_DIR.rglob("*.md"))
py_files = list(DATA_DIR.rglob("*.py"))
jsonl_files = list(DATA_DIR.rglob("*.jsonl"))

print(f"""
{'='*60}
  DOWNLOAD COMPLETE
{'='*60}
  Directory:     {DATA_DIR}
  Total files:   {len(all_files):,}
  Total size:    {total_bytes/1024/1024/1024:.2f} GB
  Readable by pipeline:
    .txt:        {len(txt_files):,}
    .md:         {len(md_files):,}
    .py:         {len(py_files):,}
    .jsonl:      {len(jsonl_files):,}

  TRAIN COMMAND:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
