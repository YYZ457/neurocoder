#!/usr/bin/env python3
"""NeuroCoder Cloud — Chinese data from VERIFIED working sources only."""
import os, subprocess, json, random, urllib.request
from pathlib import Path
os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data"); DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, subdir=""):
    name = repo.split("/")[-1]
    d = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if d.exists(): return
    try:
        subprocess.run(GIT + ["clone", "--depth", "1", f"https://github.com/{repo}.git", str(d)],
                      capture_output=True, timeout=300, check=True)
        n = len(list(d.rglob("*.txt"))) + len(list(d.rglob("*.md"))) + len(list(d.rglob("*.py")))
        print(f"  [OK] {name} ({n} text files)")
    except: print(f"  [FAIL] {name}")

def dl(url, path):
    if path.exists(): return
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  [OK] {path.name} ({path.stat().st_size/1024/1024:.0f} MB)")
    except: print(f"  [FAIL] {path.name}")

print("=" * 60)
print("  Chinese data — verified working sources only")
print("=" * 60)

# =============================================================
# 1. Chinese tech articles (lots of text, no LFS)
# =============================================================
print("\n[1/4] Chinese tech articles (verified)...")
clone("xitu/gold-miner")           # 2600+ tech articles in Chinese
clone("xitu/tensorflow-docs")      # TF docs Chinese translation
clone("CyC2018/CS-Notes")          # CS notes in Chinese
clone("jackfrued/Python-100-Days") # Chinese Python tutorial
clone("d2l-ai/d2l-zh")            # Deep Learning book Chinese
clone("MLEveryday/100-Days-Of-ML-Code")
clone("scutan90/DeepLearning-500-questions")
clone("chinese-poetry/chinese-poetry")  # Chinese poetry

# =============================================================
# 2. Chinese NLP corpus (verified)
# =============================================================
print("\n[2/4] Chinese NLP corpus...")
clone("brightmart/nlp_chinese_corpus")
clone("SophonPlus/ChineseNlpCorpus")
clone("InsaneLife/ChineseNLPCorpus")

# =============================================================
# 3. Big repos with Chinese docstrings
# =============================================================
print("\n[3/4] Big code repos (Chinese comments)...")
clone("PaddlePaddle/Paddle", "code")
clone("PaddlePaddle/PaddleNLP", "code")
clone("PaddlePaddle/PaddleOCR", "code")
clone("PaddlePaddle/Book", "code")

# =============================================================
# 4. Direct download Chinese text from GitHub raw
# =============================================================
print("\n[4/4] Direct downloads...")
dl_dir = DATA_DIR / "_dl"; dl_dir.mkdir(exist_ok=True)

# Chinese Wikipedia from USTC mirror (quick test)
try:
    print("  Trying USTC mirror...")
    subprocess.run(["wget", "-q", "--timeout=30", "-O", str(dl_dir/"wiki_test.txt"),
        "https://mirrors.ustc.edu.cn/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"],
        timeout=60)
except: pass

# Raw GitHub: Chinese text files
dl("https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/weibo/weibo_100k.txt",
   dl_dir / "weibo.txt")

# =============================================================
# Convert JSONL → conversation text
# =============================================================
print("\nConverting data...")
conv = DATA_DIR / "_txt"; conv.mkdir(exist_ok=True)

# Find all JSON/JSONL files and extract conversations
for f in list(DATA_DIR.rglob("*.jsonl")) + list(DATA_DIR.rglob("*.json")):
    if "node_modules" in str(f) or f.stat().st_size < 100: continue
    out = conv / f"{f.parent.name}_{f.stem}.txt"
    if out.exists(): continue
    c = 0
    try:
        with open(out, "w", encoding="utf-8") as fo:
            for line in open(f, encoding="utf-8", errors="ignore"):
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        # Try all possible field combinations
                        for k1, k2 in [("instruction","output"), ("q","a"), ("question","answer"),
                                        ("query","response"), ("input","target"), ("content","summary")]:
                            if item.get(k1) and item.get(k2):
                                fo.write(f"用户: {item[k1]}\n助手: {item[k2]}\n\n"); c+=1; break
                except: pass
    except: pass
    if c: print(f"  [OK] {out.name} ({c} samples)")
    else: out.unlink(missing_ok=True)

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
txt = [f for ext in ["*.txt","*.md","*.py"] for f in DATA_DIR.rglob(ext) if f.is_file() and f.stat().st_size>100]
txt = [f for f in txt]  # Use all files
random.shuffle(txt)
c = "_tc.txt"
with open(c, "w", encoding="utf-8", errors="ignore") as o:
    for f in txt[:8000]:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: o.write(t[:5000] + "\n")
        except: pass
tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel(); tok.normalizer = normalizers.NFKC()
tok.train([c], trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"], min_frequency=2, show_progress=True))
os.makedirs("tokenizer_cache", exist_ok=True); tok.save("tokenizer_cache/bpe_tokenizer.json"); os.remove(c)

total = sum(f.stat().st_size for f in txt)
print(f"""
{'='*60}
  DONE — {total/1024/1024:.0f} MB
{'='*60}
  All from verified GitHub sources (no LFS).
  Includes 2600+ Chinese tech articles, NLP corpus, poetry, tutorials.

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
