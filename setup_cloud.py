#!/usr/bin/env python3
"""NeuroCoder Cloud — downloads GBs of Chinese data. Fast sources only."""
import os, subprocess, urllib.request, json, re, random
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
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}")

def dl(url, dest):
    if dest.exists() and dest.stat().st_size > 1024: return
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {dest.name} ({dest.stat().st_size/1024/1024:.0f} MB)")
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")

def json_to_txt():
    conv = DATA_DIR / "_txt"; conv.mkdir(exist_ok=True)
    for f in list(DATA_DIR.rglob("*.jsonl")) + list(DATA_DIR.rglob("*.json")):
        if "node_modules" in str(f) or "package.json" in str(f): continue
        out = conv / f"{f.parent.name}_{f.stem}.txt"
        if out.exists(): continue
        c = 0
        try:
            with open(out, "w", encoding="utf-8") as fo:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    try:
                        item = json.loads(line); txt = " ".join(str(v) for v in item.values() if isinstance(v, str) and len(v)>5)
                        if txt: fo.write(txt+"\n"); c+=1
                    except: pass
            if not c: out.unlink()
        except: pass

print("=" * 60)
print("  Downloading Chinese data (fast sources)")
print("=" * 60)

# =============================================================
# BLOCK 1: Large Chinese corpora (verified, big)
# =============================================================
print("\n[1/5] Chinese text corpora...")
clone("brightmart/nlp_chinese_corpus")
clone("CLUEbenchmark/CLUECorpus2020")
clone("SophonPlus/ChineseNlpCorpus")
clone("InsaneLife/ChineseNLPCorpus")
clone("chinese-poetry/chinese-poetry")

# =============================================================
# BLOCK 2: Chinese tech & education (verified)
# =============================================================
print("\n[2/5] Chinese tech & education...")
clone("jackfrued/Python-100-Days")
clone("d2l-ai/d2l-zh")
clone("xitu/gold-miner")
clone("CyC2018/CS-Notes")
clone("MLEveryday/100-Days-Of-ML-Code")
clone("scutan90/DeepLearning-500-questions")

# =============================================================
# BLOCK 3: Big repos with Chinese docstrings (PaddlePaddle family)
# =============================================================
print("\n[3/5] Big repos (Chinese code + comments)...")
clone("PaddlePaddle/Paddle", "code")
clone("PaddlePaddle/PaddleNLP", "code")
clone("PaddlePaddle/PaddleOCR", "code")
clone("PaddlePaddle/PaddleDetection", "code")
clone("PaddlePaddle/PaddleSeg", "code")
clone("PaddlePaddle/PaddleGAN", "code")
clone("PaddlePaddle/PaddleRec", "code")
clone("PaddlePaddle/Book", "code")
clone("THUDM/CodeGeeX", "code")
clone("microsoft/unilm", "code")

# =============================================================
# BLOCK 4: Chinese text from NLP libraries
# =============================================================
print("\n[4/5] Chinese NLP libraries...")
clone("fxsjy/jieba")

# =============================================================
# BLOCK 5: Python code
# =============================================================
print("\n[5/5] Python code...")
for repo in ["pallets/flask", "psf/black", "pydantic/pydantic",
              "Textualize/rich", "kennethreitz/requests"]:
    clone(repo, "python")

# =============================================================
# Convert JSON → txt & Tokenizer
# =============================================================
print("\nConverting JSON...")
json_to_txt()

print("\nRetraining tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

train_files = list(DATA_DIR.rglob("*.txt")) + list(DATA_DIR.rglob("*.py")) + list(DATA_DIR.rglob("*.md"))
random.seed(42)
sample = random.sample(train_files, min(len(train_files), 60000))
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
tok.train([corpus], trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"], min_frequency=2, show_progress=True))
os.makedirs("tokenizer_cache", exist_ok=True)
tok.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(corpus)

# =============================================================
# Summary
# =============================================================
files = [f for ext in ["*.txt","*.md","*.py"] for f in DATA_DIR.rglob(ext) if f.is_file()]
total = sum(f.stat().st_size for f in files)

print(f"""
{'='*60}
  DONE — {total/1024/1024/1024:.1f} GB
{'='*60}
  Files: {len(files):,}
  Tokenizer: Chinese-aware

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
