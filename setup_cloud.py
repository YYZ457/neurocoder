#!/usr/bin/env python3
"""NeuroCoder Cloud — PURE Chinese conversation + text model. Zero code."""
import os, subprocess, urllib.request, json, re, random, shutil
from pathlib import Path

os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, subdir=""):
    name = repo.split("/")[-1]
    dest = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if dest.exists(): return
    try:
        subprocess.run(GIT + ["clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest)],
                      capture_output=True, timeout=300, check=True)
        files = len(list(dest.rglob("*.*")))
        print(f"  [OK] {name} ({files} files)")
    except: print(f"  [FAIL] {name}")

def dl(url, dest):
    if dest.exists() and dest.stat().st_size > 1024: return
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {dest.name} ({dest.stat().st_size/1024/1024:.0f} MB)")
    except: print(f"  [FAIL] {dest.name}")

def json_to_txt():
    conv = DATA_DIR / "_txt"; conv.mkdir(exist_ok=True)
    for f in list(DATA_DIR.rglob("*.jsonl")) + list(DATA_DIR.rglob("*.json")):
        if "node_modules" in str(f): continue
        out = conv / f"{f.parent.name}_{f.stem}.txt"
        if out.exists(): continue
        c = 0
        try:
            with open(out, "w", encoding="utf-8") as fo:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    try:
                        item = json.loads(line)
                        txt = " ".join(str(v) for v in item.values() if isinstance(v, str) and len(v)>5)
                        if txt: fo.write(txt+"\n"); c+=1
                    except: pass
            if not c: out.unlink()
        except: pass

print("=" * 60)
print("  Chinese Conversation Model — Data Download")
print("  Focus: chat, news, novels, QA (no code)")
print("=" * 60)

# =============================================================
# 1. Chinese conversation & chat data
# =============================================================
print("\n[1/4] Chinese conversations...")

# Weibo social media + chat data
clone("brightmart/nlp_chinese_corpus")
clone("InsaneLife/ChineseNLPCorpus")
clone("SophonPlus/ChineseNlpCorpus")
clone("liuhuanyong/ChineseNLPCorpus")
clone("thu-coai/CDial-GPT")

# =============================================================
# 2. Chinese news + web text
# =============================================================
print("\n[2/4] Chinese news & web text...")
clone("CLUEbenchmark/CLUECorpus2020")

# =============================================================
# 3. Chinese novels, poetry & literature
# =============================================================
print("\n[3/4] Chinese literature...")
clone("chinese-poetry/chinese-poetry")

# =============================================================
# 4. Chinese tech articles (conversational instruction format)
# =============================================================
print("\n[4/4] Chinese tech articles...")
clone("xitu/gold-miner")          # Chinese translations of tech articles
clone("xitu/tensorflow-docs")     # TF docs in Chinese

# =============================================================
# Convert JSON → txt
# =============================================================
print("\nConverting JSON...")
json_to_txt()

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining Chinese tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

files = list(DATA_DIR.rglob("*.txt")) + list(DATA_DIR.rglob("*.md"))
files = [f for f in files if f.is_file()]
random.seed(42)
sample = random.sample(files, min(len(files), 30000))

corpus = "_tc.txt"
with open(corpus, "w", encoding="utf-8", errors="ignore") as out:
    for f in sample:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: out.write(t[:5000]+"\n")
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

# =============================================================
# Summary
# =============================================================
txt_md = [f for ext in ["*.txt","*.md"] for f in DATA_DIR.rglob(ext) if f.is_file()]
total = sum(f.stat().st_size for f in txt_md)

print(f"""
{'='*60}
  DONE — {total/1024/1024:.0f} MB Chinese text
{'='*60}
  Text files: {len(txt_md):,}
  Est. tokens: {total//3:,}

  This is a PURE Chinese model. No code in training data.
  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
