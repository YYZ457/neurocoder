#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCoder Cloud Setup — downloads REAL Chinese data.
"""
import os, subprocess, urllib.request, json
from pathlib import Path

GIT_CMD = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def dl(url, dest):
    if dest.exists():
        print(f"  [Skip] {dest.name}")
        return
    try:
        print(f"  Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"    {dest.stat().st_size/1024/1024:.1f} MB")
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")

def clone(repo, dest_dir):
    name = repo.split("/")[-1]
    dest = Path(dest_dir) / name
    if dest.exists():
        print(f"  [Skip] {name}")
        return
    url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(GIT_CMD + ["clone", "--depth", "1", url, str(dest)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      timeout=120, check=True)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}")

def extract_jsonl_text(in_path, out_path, fields=None):
    """Convert JSONL to plain text."""
    out_path.parent.mkdir(exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for line in open(in_path, encoding="utf-8", errors="ignore"):
            try:
                item = json.loads(line)
                if fields:
                    text = " ".join(str(item.get(f, "")) for f in fields)
                else:
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                if len(text) > 10:
                    out.write(text.strip() + "\n\n")
                    count += 1
            except:
                pass
    print(f"    -> {out_path.name} ({count} lines)")

print("=" * 60)
print("  Downloading REAL Chinese data")
print("=" * 60)

# =============================================================
# 1. Chinese Wikipedia (preprocessed plain text)
# =============================================================
print("\n[1/5] Chinese Wikipedia...")
wiki_dir = DATA_DIR / "wiki"
wiki_dir.mkdir(exist_ok=True)

dl("https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.jsonl",
   wiki_dir / "wiki.jsonl")

# =============================================================
# 2. Chinese news
# =============================================================
print("\n[2/5] Chinese news corpus...")
news_dir = DATA_DIR / "news"
news_dir.mkdir(exist_ok=True)

# Try different sources
news_urls = [
    "https://hf-mirror.com/datasets/CLUE/CLUECorpus2020/resolve/main/news_sohusite_500k.jsonl",
]
for url in news_urls:
    dl(url, news_dir / "news_sohu.jsonl")

# =============================================================
# 3. Chinese conversation data from GitHub
# =============================================================
print("\n[3/5] Chinese conversation datasets...")
chat_dir = DATA_DIR / "chats"
chat_dir.mkdir(exist_ok=True)

# Clone repos with real Chinese chat data
for repo in [
    "coderchen01/Chinese_Chat_Corpus",
    "chatopera/Sentence-Corpus",
    "liuhuanyong/ChineseNLPCorpus",
]:
    clone(repo, chat_dir)

# Download BELLE instruction data
belle_dir = chat_dir / "belle"
belle_dir.mkdir(exist_ok=True)
dl("https://hf-mirror.com/datasets/BelleGroup/train_0.5M_CN/resolve/main/train_0.5M_CN.json",
   belle_dir / "belle_0.5M.json")

# =============================================================
# 4. Chinese literature
# =============================================================
print("\n[4/5] Chinese literature...")
lit_dir = DATA_DIR / "literature"
lit_dir.mkdir(exist_ok=True)

clone("chinese-poetry/chinese-poetry", lit_dir)

# Download Chinese novels
novel_dir = lit_dir / "novels"
novel_dir.mkdir(exist_ok=True)
for url, name in [
    ("https://raw.githubusercontent.com/kourgeorge/novel-chinese/master/novels/西游记.txt", "西游记.txt"),
]:
    dl(url, novel_dir / name)

# =============================================================
# 5. Python code (curated, balanced)
# =============================================================
print("\n[5/5] Python code...")
py_dir = DATA_DIR / "python"
py_dir.mkdir(exist_ok=True)

for repo in [
    "pallets/flask",
    "pydantic/pydantic",
    "psf/black",
    "Textualize/rich",
    "kennethreitz/requests",
    "pallets/click",
    "pytest-dev/pytest",
]:
    clone(repo, py_dir)

# =============================================================
# Convert everything to .txt
# =============================================================
print("\nConverting to .txt format...")
conv_dir = DATA_DIR / "_txt"
conv_dir.mkdir(exist_ok=True)

# Convert JSONL (Wikipedia, news)
for f in DATA_DIR.rglob("*.jsonl"):
    out = conv_dir / f"{f.parent.name}_{f.stem}.txt"
    extract_jsonl_text(f, out)

# Convert JSON (BELLE format: instruction, output)
for f in DATA_DIR.rglob("*.json"):
    if f.name == "package.json" or "node_modules" in str(f):
        continue
    out = conv_dir / f"{f.parent.name}_{f.stem}.txt"
    try:
        data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        count = 0
        with open(out, "w", encoding="utf-8") as fo:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        parts = []
                        for key in ["instruction", "input", "output", "q", "a", "question", "answer"]:
                            if key in item and isinstance(item[key], str):
                                parts.append(item[key])
                        text = " ".join(parts)
                        if len(text) > 10:
                            fo.write(text + "\n\n")
                            count += 1
        print(f"  [OK] {out.name} ({count} lines)")
    except:
        pass

# =============================================================
# Summary
# =============================================================
all_files = list(DATA_DIR.rglob("*.*"))
total_bytes = sum(f.stat().st_size for f in all_files if f.is_file())
txts = len(list(DATA_DIR.rglob("*.txt")))

print(f"""
{'='*60}
  DOWNLOAD COMPLETE
{'='*60}
  Directory:  chinese_data
  Size:       {total_bytes/1024/1024/1024:.2f} GB
  .txt files: {txts:,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
